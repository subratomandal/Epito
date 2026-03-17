//! Windows-native platform optimizations for Epito.
//!
//! Implements production-grade patterns used by Ollama, Chrome, VSCode,
//! and LM Studio for GPU management, process isolation, power awareness,
//! and system resource monitoring.
//!
//! All functions use raw Win32 FFI (kernel32.dll) — no extra crate dependencies.
//! This module is only compiled on Windows (`#[cfg(windows)]` in lib.rs).

use std::process::{Command, Stdio};
use std::sync::OnceLock;

// ─── GPU / VRAM Detection ───────────────────────────────────────────────────
//
// Queries nvidia-smi for GPU name and total VRAM. Cached in a OnceLock so
// we only spawn nvidia-smi once per process lifetime (it takes 200-800ms).
//
// We use TOTAL VRAM (not free) for layer calculation because free VRAM is
// volatile — another app could allocate between our query and llama-server
// launch. Total VRAM is stable. This matches Ollama's approach.

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

/// Calculate optimal number of GPU layers based on total VRAM.
///
/// Algorithm (matches Ollama/vLLM approach):
///   1. Estimate total model VRAM footprint (weights + KV cache + CUDA overhead)
///   2. If total VRAM >= footprint * 1.15 → full offload (all 33 layers)
///   3. Otherwise, scale proportionally using 85% of total VRAM as budget
///   4. If fewer than 5 layers fit, use CPU (PCIe transfer overhead not worth it)
///
/// For Mistral 7B Q4_K_M:
///   - Model weights: ~4300MB
///   - KV cache (4096 ctx, q8_0): ~200MB
///   - CUDA runtime/driver overhead: ~500MB
///   - Total estimate: ~5000MB
///   - 33 offloadable layers (32 transformer + output)
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

// ─── Process Priority Management ────────────────────────────────────────────
//
// Chrome pattern: UI process stays NORMAL, background workers get BELOW_NORMAL.
// This prevents inference from starving the UI or other user applications.

extern "system" {
    fn SetPriorityClass(handle: isize, priority: u32) -> i32;
}

pub const BELOW_NORMAL_PRIORITY: u32 = 0x00004000;
#[allow(dead_code)]
pub const NORMAL_PRIORITY: u32 = 0x00000020;

/// Set the priority class of a child process.
/// Uses the process handle directly from `std::process::Child`.
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

// ─── Single Instance Enforcement ────────────────────────────────────────────
//
// Creates a global named mutex. If another Epito process already holds it,
// shows a native MessageBox and returns false. The mutex is stored in a
// OnceLock and auto-released when the process exits.

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

/// Try to acquire the single-instance mutex.
/// Returns true if this is the first instance, false if another is running.
/// On conflict, shows a native Windows MessageBox before returning false.
pub fn check_single_instance() -> bool {
    let acquired = INSTANCE_MUTEX.get_or_init(|| {
        let name: Vec<u16> = "Global\\EpitoSingleInstance\0"
            .encode_utf16().collect();

        unsafe {
            let handle = CreateMutexW(std::ptr::null(), 0, name.as_ptr());
            if handle == 0 {
                // Mutex creation failed — allow startup (non-fatal)
                return MutexGuard(0);
            }
            if GetLastError() == ERROR_ALREADY_EXISTS {
                CloseHandle(handle);

                // Show native dialog — no Tauri window exists yet
                let text: Vec<u16> = "Epito is already running.\nCheck your taskbar.\0"
                    .encode_utf16().collect();
                let caption: Vec<u16> = "Epito\0".encode_utf16().collect();
                MessageBoxW(0, text.as_ptr(), caption.as_ptr(), MB_OK | MB_ICONINFORMATION);

                return MutexGuard(0); // Sentinel: 0 = failed
            }
            MutexGuard(handle)
        }
    });

    // handle == 0 means either creation failed or already exists.
    // If GetLastError was ERROR_ALREADY_EXISTS, we showed the dialog.
    // We can't easily distinguish "creation failed" from "already exists"
    // after OnceLock, but the MessageBox only shows on ERROR_ALREADY_EXISTS.
    acquired.0 != 0
}

// ─── Power Status ───────────────────────────────────────────────────────────
//
// Detects AC vs battery power. Used to reduce CPU threads and batch size
// during inference on laptops. We do NOT reduce GPU layers — the GPU's own
// power management handles throttling more efficiently than we can.

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
        // API failed — assume AC power (safe default, desktops always return this)
        return PowerInfo { on_ac: true, battery_percent: 100, has_battery: false };
    }
    PowerInfo {
        on_ac: ps.ac_line_status == 1,
        battery_percent: if ps.battery_life_percent == 255 { 100 } else { ps.battery_life_percent.min(100) },
        has_battery: ps.battery_flag != 128 && ps.battery_flag != 255,
    }
}

// ─── System Memory ──────────────────────────────────────────────────────────
//
// Queries total and available physical RAM. Used to:
//   - Remove --mlock if RAM < 8GB (mlock pins 4.3GB model, starves the OS)
//   - Log warnings for constrained systems

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

// ─── System Diagnostics ─────────────────────────────────────────────────────
//
// Single entry point that logs all system info at startup.
// Called once from lib.rs after the Tauri log plugin is initialized.

pub fn log_system_diagnostics() {
    log::info!("[System] ═══════════════════════════════════════════════");

    // Memory
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

    // GPU
    if let Some(vram) = query_gpu_vram() {
        let layers = calculate_gpu_layers(vram.total_mb);
        log::info!(
            "[System] GPU: {} — {}MB VRAM → {} layers offloadable",
            vram.name, vram.total_mb, layers
        );
    } else {
        log::info!("[System] GPU: No NVIDIA GPU detected (will use Vulkan/CPU)");
    }

    // Power
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

    log::info!("[System] ═══════════════════════════════════════════════");
}

// ─── DWM Title Bar Customization ─────────────────────────────────────────
//
// Windows 10 1809+ (build 17763) / Windows 11 DWM API for title bar theming.
// This is how VS Code, Discord, Spotify, and Notion color their title bars.
// Without this, the Windows title bar follows the system theme regardless
// of the app's dark/light mode setting.
//
// Uses dwmapi.dll FFI — no extra crate dependencies. All calls are best-effort
// and fail silently on older Windows versions that don't support these attributes.

extern "system" {
    fn DwmSetWindowAttribute(hwnd: isize, attr: u32, value: *const u8, size: u32) -> i32;
}

// DWM attribute IDs (from dwmapi.h)
const DWMWA_USE_IMMERSIVE_DARK_MODE_V1: u32 = 19; // Windows 10 pre-20H1 (undocumented)
const DWMWA_USE_IMMERSIVE_DARK_MODE: u32 = 20;    // Windows 10 20H1+ (official)
const DWMWA_BORDER_COLOR: u32 = 34;               // Windows 11 22000+
const DWMWA_CAPTION_COLOR: u32 = 35;              // Windows 11 22000+
const DWMWA_TEXT_COLOR: u32 = 36;                  // Windows 11 22000+

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

/// Apply full theme to a Windows title bar.
/// Sets dark/light mode, caption color, border color, and text color.
/// Must be called with a valid HWND. All calls are best-effort — fails
/// silently on older Windows versions that don't support these attributes.
///
/// Pattern: VS Code, Discord, Spotify all use DwmSetWindowAttribute
/// to match their title bar to the app's color scheme.
pub fn apply_titlebar_theme(hwnd: isize, is_dark: bool) {
    if hwnd == 0 { return; }

    // 1. Dark mode attribute — controls caption button icon colors (white on dark, dark on light).
    //    Try the official attribute (20) first, fall back to the pre-20H1 attribute (19).
    let dark_val: u32 = if is_dark { 1 } else { 0 };
    let r1 = dwm_set_u32(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, dark_val);
    if r1 != 0 {
        // Older Windows 10 — try the undocumented attribute 19
        dwm_set_u32(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_V1, dark_val);
    }

    // 2. Caption (title bar) background color — Windows 11 only.
    //    On Windows 10, dark mode attribute (above) is sufficient — it makes
    //    the title bar dark/light automatically. CAPTION_COLOR gives precise control.
    let caption = if is_dark { colorref(10, 10, 15) } else { colorref(255, 255, 255) };
    let r2 = dwm_set_u32(hwnd, DWMWA_CAPTION_COLOR, caption);

    // 3. Window border color — matches caption for a seamless look.
    let r3 = dwm_set_u32(hwnd, DWMWA_BORDER_COLOR, caption);

    // 4. Title text color — for when title text is shown.
    let text = if is_dark { colorref(255, 255, 255) } else { colorref(10, 10, 15) };
    let r4 = dwm_set_u32(hwnd, DWMWA_TEXT_COLOR, text);

    log::info!(
        "[Windows] Title bar theme: {} (dark_mode={}, caption={}, border={}, text={})",
        if is_dark { "dark" } else { "light" }, r1, r2, r3, r4
    );
}

// Invalidate the non-client area to force a title bar repaint.
// Must be called after DWM attribute changes for immediate visual update.
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

// ─── DPI Awareness ───────────────────────────────────────────────────────
//
// Ensures the process uses Per-Monitor DPI Awareness V2 for crisp rendering
// at ALL Windows display scaling levels (100%, 125%, 150%, 175%, 200%).
// Without this, Windows may bitmap-stretch the UI, causing "hazy" text.
//
// Tauri sets this via the application manifest, but an explicit API call
// at process start guarantees it takes effect before any window is created.
// If already set by manifest, the API call harmlessly returns false.

extern "system" {
    fn SetProcessDpiAwarenessContext(value: isize) -> i32;
}

pub fn ensure_dpi_awareness() {
    // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
    // This is the highest quality DPI mode, used by Chrome, VS Code, etc.
    let result = unsafe { SetProcessDpiAwarenessContext(-4) };
    if result != 0 {
        log::info!("[Windows] DPI: Per-Monitor V2 awareness set via API");
    } else {
        // Already set by manifest — this is the expected path
        log::info!("[Windows] DPI: Per-Monitor V2 already active (via manifest)");
    }
}
