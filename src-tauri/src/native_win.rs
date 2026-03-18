//! Windows-native platform APIs for GPU management, process isolation,
//! power awareness, and system resource monitoring.
//! All functions use raw Win32 FFI — no extra crate dependencies.

use std::process::{Command, Stdio};
use std::sync::OnceLock;

// -- GPU / VRAM Detection --
// Uses total VRAM (not free) because free VRAM is volatile between query
// and llama-server launch. Cached via OnceLock (nvidia-smi takes 200-800ms).

#[derive(Debug, Clone)]
pub struct GpuVramInfo {
    pub name: String,
    pub total_mb: u64,
}

static VRAM_CACHE: OnceLock<Option<GpuVramInfo>> = OnceLock::new();

pub fn query_gpu_vram() -> Option<&'static GpuVramInfo> {
    VRAM_CACHE.get_or_init(|| {
        use std::os::windows::process::CommandExt;
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .creation_flags(0x08000000) // CREATE_NO_WINDOW
            .output()
            .ok()?;

        if !output.status.success() { return None; }

        let text = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = text.trim().splitn(2, ',').map(|s| s.trim()).collect();
        if parts.len() < 2 { return None; }

        let total_mb = parts[1].parse::<u64>().ok()?;
        let name = parts[0].to_string();

        log::info!("[GPU] VRAM detected: {} — {}MB total", name, total_mb);
        Some(GpuVramInfo { name, total_mb })
    }).as_ref()
}

/// Calculate GPU layers based on total VRAM (Ollama/vLLM approach).
/// Full offload at 115% of footprint, proportional below that, CPU if <5 layers fit.
pub fn calculate_gpu_layers(total_vram_mb: u64) -> u32 {
    const MODEL_FOOTPRINT_MB: u64 = 5000; // weights + KV cache + CUDA overhead
    const MAX_LAYERS: u32 = 33;           // Mistral 7B: 32 transformer + 1 output
    const MIN_WORTHWHILE: u32 = 5;        // Below this, CPU is faster (PCIe bottleneck)

    // Full offload: VRAM >= footprint * 1.15 (15% headroom)
    if total_vram_mb >= MODEL_FOOTPRINT_MB * 115 / 100 {
        log::info!(
            "[GPU] {}MB VRAM >= {}MB needed — full offload ({} layers)",
            total_vram_mb, MODEL_FOOTPRINT_MB * 115 / 100, MAX_LAYERS
        );
        return MAX_LAYERS;
    }

    // Partial offload: use 85% of total VRAM as budget
    let budget_mb = total_vram_mb * 85 / 100;
    let per_layer_mb = MODEL_FOOTPRINT_MB / MAX_LAYERS as u64; // ~151MB/layer

    if per_layer_mb == 0 {
        return MAX_LAYERS;
    }

    let layers = (budget_mb / per_layer_mb) as u32;
    let layers = layers.min(MAX_LAYERS);

    if layers < MIN_WORTHWHILE {
        log::info!(
            "[GPU] {}MB VRAM → only {} layers fit (need ≥{}) — using CPU instead",
            total_vram_mb, layers, MIN_WORTHWHILE
        );
        return 0;
    }

    log::info!(
        "[GPU] {}MB VRAM → {} of {} layers ({}MB budget, {}MB/layer)",
        total_vram_mb, layers, MAX_LAYERS, budget_mb, per_layer_mb
    );
    layers
}

// -- Process Priority --
// Chrome pattern: UI stays NORMAL, inference workers get BELOW_NORMAL.

extern "system" {
    fn SetPriorityClass(handle: isize, priority: u32) -> i32;
}

pub const BELOW_NORMAL_PRIORITY: u32 = 0x00004000;
#[allow(dead_code)]
pub const NORMAL_PRIORITY: u32 = 0x00000020;

/// Set the priority class of a child process.
pub fn set_process_priority(child: &std::process::Child, priority: u32) {
    use std::os::windows::io::AsRawHandle;
    let handle = child.as_raw_handle() as isize;
    if handle == 0 { return; }

    let ok = unsafe { SetPriorityClass(handle, priority) };
    let label = match priority {
        0x00004000 => "BELOW_NORMAL",
        0x00000020 => "NORMAL",
        _ => "CUSTOM",
    };
    if ok != 0 {
        log::info!("[Windows] PID {} priority → {}", child.id(), label);
    } else {
        log::warn!("[Windows] Failed to set PID {} priority to {}", child.id(), label);
    }
}

// -- Single Instance --
// Global named mutex — shows MessageBox if already held by another process.

extern "system" {
    fn CreateMutexW(attrs: *const u8, initial: i32, name: *const u16) -> isize;
    fn GetLastError() -> u32;
    fn CloseHandle(handle: isize) -> i32;
    fn MessageBoxW(hwnd: isize, text: *const u16, caption: *const u16, flags: u32) -> i32;
}

const ERROR_ALREADY_EXISTS: u32 = 183;
const MB_OK: u32 = 0x00000000;
const MB_ICONINFORMATION: u32 = 0x00000040;

struct MutexGuard(isize);
unsafe impl Send for MutexGuard {}
unsafe impl Sync for MutexGuard {}
impl Drop for MutexGuard {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe { CloseHandle(self.0); }
        }
    }
}

static INSTANCE_MUTEX: OnceLock<MutexGuard> = OnceLock::new();

/// Returns true if this is the first instance, false (with MessageBox) if another is running.
pub fn check_single_instance() -> bool {
    let acquired = INSTANCE_MUTEX.get_or_init(|| {
        let name: Vec<u16> = "Global\\EpitoSingleInstance\0"
            .encode_utf16().collect();

        unsafe {
            let handle = CreateMutexW(std::ptr::null(), 0, name.as_ptr());
            if handle == 0 {
                return MutexGuard(0);
            }
            if GetLastError() == ERROR_ALREADY_EXISTS {
                CloseHandle(handle);
                let text: Vec<u16> = "Epito is already running.\nCheck your taskbar.\0"
                    .encode_utf16().collect();
                let caption: Vec<u16> = "Epito\0".encode_utf16().collect();
                MessageBoxW(0, text.as_ptr(), caption.as_ptr(), MB_OK | MB_ICONINFORMATION);

                return MutexGuard(0); // Sentinel: 0 = failed
            }
            MutexGuard(handle)
        }
    });

    acquired.0 != 0
}

// -- Power Status --

#[repr(C)]
struct SystemPowerStatus {
    ac_line_status: u8,        // 0=offline, 1=online, 255=unknown
    battery_flag: u8,          // 128=no battery
    battery_life_percent: u8,  // 0-100, 255=unknown
    system_status_flag: u8,
    battery_life_time: u32,
    battery_full_life_time: u32,
}

extern "system" {
    fn GetSystemPowerStatus(status: *mut SystemPowerStatus) -> i32;
}

#[derive(Debug, Clone)]
pub struct PowerInfo {
    pub on_ac: bool,
    pub battery_percent: u8,
    pub has_battery: bool,
}

pub fn get_power_status() -> PowerInfo {
    let mut ps: SystemPowerStatus = unsafe { std::mem::zeroed() };
    let ok = unsafe { GetSystemPowerStatus(&mut ps) };
    if ok == 0 {
        return PowerInfo { on_ac: true, battery_percent: 100, has_battery: false };
    }
    PowerInfo {
        on_ac: ps.ac_line_status == 1,
        battery_percent: if ps.battery_life_percent == 255 { 100 } else { ps.battery_life_percent.min(100) },
        has_battery: ps.battery_flag != 128 && ps.battery_flag != 255,
    }
}

// -- System Memory --

#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

extern "system" {
    fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub available_mb: u64,
    pub usage_percent: u32,
}

pub fn get_system_memory() -> MemoryInfo {
    let mut ms: MemoryStatusEx = unsafe { std::mem::zeroed() };
    ms.dw_length = std::mem::size_of::<MemoryStatusEx>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut ms) };
    if ok == 0 {
        return MemoryInfo { total_mb: 0, available_mb: 0, usage_percent: 0 };
    }
    MemoryInfo {
        total_mb: ms.ull_total_phys / (1024 * 1024),
        available_mb: ms.ull_avail_phys / (1024 * 1024),
        usage_percent: ms.dw_memory_load,
    }
}

// -- System Diagnostics --

pub fn log_system_diagnostics() {
    let mem = get_system_memory();
    if mem.total_mb > 0 {
        log::info!(
            "[System] RAM: {:.1}GB total, {:.1}GB available ({}% used)",
            mem.total_mb as f64 / 1024.0,
            mem.available_mb as f64 / 1024.0,
            mem.usage_percent
        );
        if mem.total_mb < 8192 {
            log::warn!(
                "[System] Low RAM ({:.1}GB) — --mlock will be disabled to prevent OS starvation",
                mem.total_mb as f64 / 1024.0
            );
        }
    }

    if let Some(vram) = query_gpu_vram() {
        let layers = calculate_gpu_layers(vram.total_mb);
        log::info!(
            "[System] GPU: {} — {}MB VRAM → {} layers offloadable",
            vram.name, vram.total_mb, layers
        );
    } else {
        log::info!("[System] GPU: No NVIDIA GPU detected (will use Vulkan/CPU)");
    }

    let power = get_power_status();
    if power.has_battery {
        log::info!(
            "[System] Power: {} ({}%)",
            if power.on_ac { "AC power" } else { "Battery" },
            power.battery_percent
        );
        if !power.on_ac {
            log::info!("[System] Battery mode → reducing inference threads and batch size");
        }
    } else {
        log::info!("[System] Power: AC (desktop)");
    }
}

// -- DWM Title Bar --
// DWM API for title bar theming (Win10 1809+ / Win11).
// Without this, Windows title bar ignores the app's dark/light setting.

extern "system" {
    fn DwmSetWindowAttribute(hwnd: isize, attr: u32, value: *const u8, size: u32) -> i32;
}

const DWMWA_USE_IMMERSIVE_DARK_MODE_V1: u32 = 19; // Win10 pre-20H1 (undocumented)
const DWMWA_USE_IMMERSIVE_DARK_MODE: u32 = 20;    // Win10 20H1+
const DWMWA_BORDER_COLOR: u32 = 34;               // Win11 22000+
const DWMWA_CAPTION_COLOR: u32 = 35;              // Win11 22000+
const DWMWA_TEXT_COLOR: u32 = 36;                  // Win11 22000+

/// Win32 COLORREF format: 0x00BBGGRR
fn colorref(r: u8, g: u8, b: u8) -> u32 {
    (b as u32) << 16 | (g as u32) << 8 | (r as u32)
}

fn dwm_set_u32(hwnd: isize, attr: u32, value: u32) -> i32 {
    unsafe {
        DwmSetWindowAttribute(
            hwnd,
            attr,
            &value as *const u32 as *const u8,
            std::mem::size_of::<u32>() as u32,
        )
    }
}

/// Apply dark/light theme to title bar. Best-effort — silently ignored on older Windows.
pub fn apply_titlebar_theme(hwnd: isize, is_dark: bool) {
    if hwnd == 0 { return; }

    // Try official attr 20, fall back to undocumented attr 19 for pre-20H1
    let dark_val: u32 = if is_dark { 1 } else { 0 };
    let r1 = dwm_set_u32(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, dark_val);
    if r1 != 0 {
        dwm_set_u32(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_V1, dark_val);
    }

    let caption = if is_dark { colorref(10, 10, 15) } else { colorref(255, 255, 255) };
    let r2 = dwm_set_u32(hwnd, DWMWA_CAPTION_COLOR, caption);
    let r3 = dwm_set_u32(hwnd, DWMWA_BORDER_COLOR, caption);

    let text = if is_dark { colorref(255, 255, 255) } else { colorref(10, 10, 15) };
    let r4 = dwm_set_u32(hwnd, DWMWA_TEXT_COLOR, text);

    log::info!(
        "[Windows] Title bar theme: {} (dark_mode={}, caption={}, border={}, text={})",
        if is_dark { "dark" } else { "light" }, r1, r2, r3, r4
    );
}

// Force title bar repaint after DWM attribute changes
extern "system" {
    fn SetWindowPos(hwnd: isize, after: isize, x: i32, y: i32, cx: i32, cy: i32, flags: u32) -> i32;
}
const SWP_NOMOVE: u32 = 0x0002;
const SWP_NOSIZE: u32 = 0x0001;
const SWP_NOZORDER: u32 = 0x0004;
const SWP_FRAMECHANGED: u32 = 0x0020;

pub fn force_titlebar_redraw(hwnd: isize) {
    if hwnd == 0 { return; }
    unsafe {
        SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
    }
}

// -- DPI Awareness --
// Explicit API call guarantees Per-Monitor V2 before any window creation.
// Tauri's manifest also sets this, but the API call is a safety net.

extern "system" {
    fn SetProcessDpiAwarenessContext(value: isize) -> i32;
}

pub fn ensure_dpi_awareness() {
    // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
    let result = unsafe { SetProcessDpiAwarenessContext(-4) };
    if result != 0 {
        log::info!("[Windows] DPI: Per-Monitor V2 set via API");
    } else {
        log::info!("[Windows] DPI: Per-Monitor V2 already active (via manifest)");
    }
}
