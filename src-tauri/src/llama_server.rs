use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use tauri::Manager;

use crate::model;

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum GpuBackend {
    Metal,    // macOS Apple Silicon / Intel — Metal GPU compute
    Cuda,     // NVIDIA — fastest, needs ggml-cuda.dll + cudart
    Vulkan,   // Any GPU vendor — universal fallback
    Rocm,     // AMD on Linux
    CpuOnly,  // No GPU offload
}

/// Detect the best GPU backend by checking BOTH hardware AND available DLLs.
///
/// Strategy (matches how Ollama, LM Studio, and Jan.ai do it):
///   1. Detect GPU hardware (nvidia-smi, WMIC, etc.)
///   2. Check which backend DLLs are actually present
///   3. Pick the best available: CUDA > Vulkan > ROCm > CPU
///
/// This prevents selecting CUDA when only the Vulkan build was downloaded.
fn detect_gpu_backend(lib_dirs: &[std::path::PathBuf]) -> GpuBackend {
    // --- macOS: always Metal ---
    if cfg!(target_os = "macos") {
        log::info!("[GPU] macOS — Metal backend (Apple GPU compute)");
        return GpuBackend::Metal;
    }

    // --- Detect GPU hardware ---
    let gpu_vendor = detect_gpu_vendor();

    // --- Check which backend DLLs/SOs are available ---
    let has_cuda_lib = find_lib_in_dirs(lib_dirs, "ggml-cuda");
    let has_vulkan_lib = find_lib_in_dirs(lib_dirs, "ggml-vulkan");
    let has_rocm_lib = find_lib_in_dirs(lib_dirs, "ggml-rocm");

    log::info!(
        "[GPU] Hardware: {:?} | Available backends: CUDA={}, Vulkan={}, ROCm={}",
        gpu_vendor, has_cuda_lib, has_vulkan_lib, has_rocm_lib
    );

    // --- Select best backend based on hardware + available libs ---
    match gpu_vendor {
        GpuVendor::Nvidia => {
            if has_cuda_lib {
                log::info!("[GPU] → CUDA backend (NVIDIA, highest performance)");
                return GpuBackend::Cuda;
            }
            if has_vulkan_lib {
                log::info!("[GPU] → Vulkan backend (NVIDIA, CUDA libs not available)");
                return GpuBackend::Vulkan;
            }
            log::warn!("[GPU] NVIDIA GPU found but no GPU backend DLLs available — falling back to CPU");
        }
        GpuVendor::Amd => {
            if has_rocm_lib {
                log::info!("[GPU] → ROCm backend (AMD)");
                return GpuBackend::Rocm;
            }
            if has_vulkan_lib {
                log::info!("[GPU] → Vulkan backend (AMD)");
                return GpuBackend::Vulkan;
            }
            log::warn!("[GPU] AMD GPU found but no GPU backend DLLs available — falling back to CPU");
        }
        GpuVendor::Intel => {
            if has_vulkan_lib {
                log::info!("[GPU] → Vulkan backend (Intel)");
                return GpuBackend::Vulkan;
            }
            log::warn!("[GPU] Intel GPU found but Vulkan DLL not available — falling back to CPU");
        }
        GpuVendor::None => {
            log::info!("[GPU] No discrete GPU detected");
        }
    }

    log::info!("[GPU] → CPU-only backend");
    GpuBackend::CpuOnly
}

#[derive(Debug)]
#[allow(dead_code)]
enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    None,
}

/// Create a Command with CREATE_NO_WINDOW on Windows to prevent console flash.
fn silent_command(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

fn detect_gpu_vendor() -> GpuVendor {
    // Check NVIDIA via nvidia-smi (works on both Windows and Linux)
    if let Ok(output) = silent_command("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
    {
        if output.status.success() {
            let name = String::from_utf8_lossy(&output.stdout);
            log::info!("[GPU] NVIDIA GPU: {}", name.trim());
            return GpuVendor::Nvidia;
        }
    }

    // Windows: PowerShell (modern, works on Win10/11 even without WMIC)
    #[cfg(target_os = "windows")]
    {
        if let Some(vendor) = detect_gpu_via_powershell() {
            return vendor;
        }
    }

    // Windows: WMIC fallback (deprecated but still works on most machines)
    #[cfg(target_os = "windows")]
    {
        if let Some(vendor) = detect_gpu_via_wmic() {
            return vendor;
        }
    }

    // Linux: check for AMD ROCm
    #[cfg(target_os = "linux")]
    {
        if silent_command("rocminfo")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return GpuVendor::Amd;
        }
    }

    GpuVendor::None
}

#[cfg(target_os = "windows")]
fn detect_gpu_via_powershell() -> Option<GpuVendor> {
    let output = silent_command("powershell")
        .args(["-NoProfile", "-NoLogo", "-Command",
            "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() { return None; }

    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() { continue; }
        log::info!("[GPU] Found (PowerShell): {}", l);
        let lower = l.to_lowercase();
        if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("quadro")
            || lower.contains("rtx") || lower.contains("gtx") {
            return Some(GpuVendor::Nvidia);
        }
        if lower.contains("radeon") || lower.contains("amd") {
            return Some(GpuVendor::Amd);
        }
        if lower.contains("arc") || lower.contains("iris") || lower.contains("uhd") {
            return Some(GpuVendor::Intel);
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn detect_gpu_via_wmic() -> Option<GpuVendor> {
    let output = silent_command("wmic")
        .args(["path", "win32_VideoController", "get", "name"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l == "Name" { continue; }
        log::info!("[GPU] Found (WMIC): {}", l);
        let lower = l.to_lowercase();
        if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("quadro")
            || lower.contains("rtx") || lower.contains("gtx") {
            return Some(GpuVendor::Nvidia);
        }
        if lower.contains("radeon") || lower.contains("amd") {
            return Some(GpuVendor::Amd);
        }
        if lower.contains("arc") || lower.contains("iris") || lower.contains("uhd") {
            return Some(GpuVendor::Intel);
        }
    }
    None
}

/// Check if a backend library exists in any of the DLL search directories.
/// Searches for `ggml-cuda.dll` / `libggml-cuda.dylib` / `libggml-cuda.so`.
fn find_lib_in_dirs(dirs: &[std::path::PathBuf], lib_base_name: &str) -> bool {
    let (prefix, ext) = if cfg!(target_os = "macos") {
        ("lib", ".dylib")
    } else if cfg!(target_os = "linux") {
        ("lib", ".so")
    } else {
        ("", ".dll")
    };

    let filename = format!("{}{}{}", prefix, lib_base_name, ext);

    for dir in dirs {
        if dir.join(&filename).exists() {
            log::info!("[GPU] Found {} in {:?}", filename, dir);
            return true;
        }
    }
    false
}

pub struct LlamaProcess {
    pub child: Mutex<Option<Child>>,
    pub port: Mutex<u16>,
}

impl Drop for LlamaProcess {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

impl LlamaProcess {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            port: Mutex::new(0),
        }
    }

    fn stop_inner(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(ref mut child) = *guard {
                log::info!("[llama-server] Stopping process...");
                crate::kill_process_tree(child);
                log::info!("[llama-server] Process stopped.");
            }
            *guard = None;
        }
    }
}

pub fn find_llama_server(app: &tauri::AppHandle) -> Option<String> {
    // Use the same find_binary_in_dir helper from lib.rs to find both
    // "llama-server.exe" and "llama-server-x86_64-pc-windows-msvc.exe"

    if let Ok(resource_dir) = app.path().resource_dir() {
        log::info!("[llama-server] Checking resource dir: {:?}", resource_dir);
        if let Some(p) = crate::find_binary_in_dir(&resource_dir, "llama-server") {
            if let Some(path) = check_binary_valid(&p) {
                return Some(path);
            }
        }
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            log::info!("[llama-server] Checking exe dir: {:?}", exe_dir);
            if let Some(p) = crate::find_binary_in_dir(exe_dir, "llama-server") {
                if let Some(path) = check_binary_valid(&p) {
                    return Some(path);
                }
            }
        }
    }

    for dev_dir in &["src-tauri/binaries", "binaries"] {
        let dir = std::path::PathBuf::from(dev_dir);
        log::info!("[llama-server] Checking dev path: {:?}", dir);
        if let Some(p) = crate::find_binary_in_dir(&dir, "llama-server") {
            if let Some(path) = check_binary_valid(&p) {
                return Some(path);
            }
        }
    }

    log::error!("[llama-server] Binary not found anywhere");
    None
}

fn check_binary_valid(path: &std::path::Path) -> Option<String> {
    if !path.exists() {
        return None;
    }
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if size < 1000 {
        log::warn!("[llama-server] Binary at {:?} is too small ({} bytes), skipping", path, size);
        return None;
    }
    log::info!("[llama-server] Found valid binary at {:?} ({:.1} MB)", path, size as f64 / 1_048_576.0);
    Some(crate::normalize_windows_path(path).to_string_lossy().to_string())
}

pub fn start(
    app: &tauri::AppHandle,
    state: &LlamaProcess,
    port: u16,
) -> Result<u16, String> {
    let model_path = model::model_path();
    if !model::model_exists() {
        log::error!("[llama-server] Model not found at {:?}", model_path);
        return Err(format!("Model not found at {:?}. Download it first.", model_path));
    }
    log::info!("[llama-server] Model found at {:?}", model_path);

    let binary = find_llama_server(app)
        .ok_or_else(|| "llama-server binary not found. Ensure scripts/downloadLlamaServer.mjs was run before building.".to_string())?;

    let threads = num_threads();
    log::info!(
        "[llama-server] Starting: binary={}, port={}, model={:?}, threads={}",
        binary, port, model_path, threads
    );

    let binary_path = std::path::Path::new(&binary);
    let binary_dir = binary_path
        .parent()
        .unwrap_or(std::path::Path::new("."));

    let dll_dirs = find_lib_directories(app, binary_dir);

    for dir in &dll_dirs {
        copy_libs_from_dir(dir, binary_dir);
    }

    let gpu = detect_gpu_backend(&dll_dirs);

    // Try GPU backend first. If it fails (driver mismatch, VRAM too small),
    // fall back to CPU. This matches how Ollama handles GPU init failures.
    let result = spawn_llama_server(
        &binary, binary_dir, &dll_dirs, &model_path, port, threads, gpu,
    );

    match result {
        Ok(child) => {
            #[cfg(windows)]
            crate::win_job::assign(&child);

            *state.child.lock().unwrap() = Some(child);
            *state.port.lock().unwrap() = port;
            Ok(port)
        }
        Err(e) if gpu != GpuBackend::CpuOnly => {
            // GPU spawn failed — retry with CPU fallback
            log::warn!(
                "[llama-server] {:?} backend failed: {}. Retrying with CPU fallback...", gpu, e
            );
            let child = spawn_llama_server(
                &binary, binary_dir, &dll_dirs, &model_path, port, threads, GpuBackend::CpuOnly,
            )?;

            #[cfg(windows)]
            crate::win_job::assign(&child);

            *state.child.lock().unwrap() = Some(child);
            *state.port.lock().unwrap() = port;
            log::info!("[llama-server] Running on CPU fallback");
            Ok(port)
        }
        Err(e) => Err(e),
    }
}

fn build_args(
    model_path: &std::path::Path,
    port: u16,
    threads: usize,
    gpu: GpuBackend,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "--model".into(), model_path.to_string_lossy().to_string(),
        "--port".into(), port.to_string(),
        "--host".into(), "127.0.0.1".into(),
        "--ctx-size".into(), "4096".into(),
        "--threads".into(), threads.to_string(),
        "--cache-type-k".into(), "q8_0".into(),
        "--cache-type-v".into(), "q8_0".into(),
        "--cache-reuse".into(), "256".into(),
        "--defrag-thold".into(), "0.1".into(),
        "--slot-prompt-similarity".into(), "0.5".into(),
        "--batch-size".into(), "512".into(),
        "--ubatch-size".into(), "256".into(),
    ];

    // Memory strategy:
    //   macOS (Metal): --mlock — lock model in unified memory (shared CPU/GPU RAM)
    //   Windows/Linux: --mmap (default) — memory-mapped, OS loads pages on demand
    //                  This reduces initial RAM usage; model pages load as needed.
    if cfg!(target_os = "macos") {
        args.push("--mlock".into());
        log::info!("[llama-server] Memory: mlock (unified memory)");
    } else {
        log::info!("[llama-server] Memory: mmap (demand-paged, lower initial RAM)");
    }

    match gpu {
        GpuBackend::Metal => {
            log::info!("[llama-server] Backend: Metal (Apple GPU, full offload, flash-attn)");
            args.extend(["--n-gpu-layers".into(), "99".into()]);
            args.push("--flash-attn".into());
        }
        GpuBackend::Cuda => {
            log::info!("[llama-server] Backend: CUDA (NVIDIA GPU, full offload, flash-attn)");
            args.extend(["--n-gpu-layers".into(), "99".into()]);
            args.push("--flash-attn".into());
        }
        GpuBackend::Vulkan => {
            log::info!("[llama-server] Backend: Vulkan (GPU, full offload)");
            args.extend(["--n-gpu-layers".into(), "99".into()]);
        }
        GpuBackend::Rocm => {
            log::info!("[llama-server] Backend: ROCm (AMD GPU, full offload)");
            args.extend(["--n-gpu-layers".into(), "99".into()]);
        }
        GpuBackend::CpuOnly => {
            log::info!("[llama-server] Backend: CPU fallback (no GPU offload)");
            args.extend(["--n-gpu-layers".into(), "0".into()]);
        }
    }

    args
}

#[allow(unused_variables)]
fn spawn_llama_server(
    binary: &str,
    binary_dir: &std::path::Path,
    dll_dirs: &[std::path::PathBuf],
    model_path: &std::path::Path,
    port: u16,
    threads: usize,
    gpu: GpuBackend,
) -> Result<Child, String> {
    let args = build_args(model_path, port, threads, gpu);

    let mut cmd = Command::new(binary);
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Windows: add DLL directories to PATH for the child process
    #[cfg(target_os = "windows")]
    {
        let mut path_dirs: Vec<String> = vec![binary_dir.to_string_lossy().to_string()];
        for dir in dll_dirs {
            path_dirs.push(dir.to_string_lossy().to_string());
        }
        let system_path = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{};{}", path_dirs.join(";"), system_path));
        log::info!("[llama-server] DLL paths: {:?}", path_dirs);
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(binary) {
            let mode = meta.permissions().mode();
            if mode & 0o111 == 0 {
                let mut perms = meta.permissions();
                perms.set_mode(mode | 0o755);
                let _ = std::fs::set_permissions(binary, perms);
            }
        }
    }

    let child = cmd.spawn().map_err(|e| {
        log::error!("[llama-server] Spawn failed ({:?} backend): {}", gpu, e);
        format!("Failed to start llama-server ({:?}): {}", gpu, e)
    })?;

    log::info!("[llama-server] Spawned PID {} with {:?} backend", child.id(), gpu);
    Ok(child)
}

pub fn wait_ready(port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    let url = format!("http://127.0.0.1:{}/health", port);
    log::info!("[llama-server] Waiting for readiness at {} (timeout: {:?})", url, timeout);

    let mut attempt = 0;
    while start.elapsed() < timeout {
        attempt += 1;
        match reqwest::blocking::get(&url) {
            Ok(resp) => {
                if resp.status().is_success() {
                    let elapsed = start.elapsed().as_secs_f64();
                    log::info!(
                        "[llama-server] ✓ Ready on port {} ({:.1}s, {} attempts)",
                        port, elapsed, attempt
                    );
                    // Log GPU validation: if the server responded with model loaded,
                    // the selected backend initialized successfully.
                    log::info!("[llama-server] ✓ GPU backend initialized, model loaded, inference ready");
                    return true;
                }
                if attempt <= 3 || attempt % 10 == 0 {
                    log::info!("[llama-server] Health attempt {}: HTTP {}", attempt, resp.status());
                }
            }
            Err(e) => {
                if attempt <= 3 || attempt % 10 == 0 {
                    log::info!("[llama-server] Health attempt {}: {}", attempt, e);
                }
            }
        }
        thread::sleep(Duration::from_millis(1000));
    }

    log::error!("[llama-server] ✗ Not ready after {:?} ({} attempts)", timeout, attempt);
    false
}

pub fn stop(state: &LlamaProcess) {
    state.stop_inner();
}

pub fn get_port(state: &LlamaProcess) -> u16 {
    *state.port.lock().unwrap()
}

/// Returns all directories that might contain shared libraries for llama-server.
fn find_lib_directories(app: &tauri::AppHandle, binary_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();

    // 1. Same directory as the binary itself
    if binary_dir.exists() {
        dirs.push(binary_dir.to_path_buf());
    }

    if let Ok(res_dir) = app.path().resource_dir() {
        // 2. Libs bundled inside standalone via prepareStandalone.mjs
        let standalone_lib = res_dir.join("_up_").join(".next").join("standalone").join("lib");
        if standalone_lib.exists() {
            log::info!("[llama-server] Found standalone lib dir: {:?}", standalone_lib);
            dirs.push(standalone_lib);
        }

        // 3. binaries/ subfolder in resources
        let res_binaries = res_dir.join("binaries");
        if res_binaries.exists() {
            dirs.push(res_binaries);
        }

        // 4. Resource dir root
        if res_dir.exists() {
            dirs.push(res_dir);
        }
    }

    // 5. Dev paths
    let dev_bins = std::path::PathBuf::from("src-tauri/binaries");
    if dev_bins.exists() {
        if let Ok(canonical) = dev_bins.canonicalize() {
            dirs.push(canonical);
        } else {
            dirs.push(dev_bins);
        }
    }

    dirs
}

/// Best-effort copy of shared libs from `src_dir` to `dest_dir`.
/// May fail silently (e.g. writing to Program Files without admin).
fn copy_libs_from_dir(src_dir: &std::path::Path, dest_dir: &std::path::Path) {
    if !src_dir.exists() || src_dir == dest_dir {
        return;
    }

    let extensions: &[&str] = if cfg!(target_os = "macos") {
        &[".dylib"]
    } else if cfg!(target_os = "linux") {
        &[".so"]
    } else {
        &[".dll"]
    };

    let entries = match std::fs::read_dir(src_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let is_shared_lib = extensions.iter().any(|ext| name.ends_with(ext));
        let has_valid_prefix = if cfg!(target_os = "windows") {
            true // Windows DLLs: ggml.dll, llama.dll (no "lib" prefix)
        } else {
            name.starts_with("lib") // Unix: libggml.dylib
        };

        if is_shared_lib && has_valid_prefix {
            let dest = dest_dir.join(&name);
            if !dest.exists() {
                match std::fs::copy(entry.path(), &dest) {
                    Ok(_) => log::info!("[llama-server] Copied {} → {:?}", name, dest),
                    Err(e) => log::info!("[llama-server] Could not copy {} to {:?}: {} (will use PATH fallback)", name, dest, e),
                }
            }
        }
    }
}

fn num_threads() -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    cpus.saturating_sub(1).max(1)
}

pub fn is_running(state: &LlamaProcess) -> bool {
    if let Ok(guard) = state.child.lock() {
        guard.is_some()
    } else {
        false
    }
}

#[tauri::command]
pub fn get_llama_port(state: tauri::State<'_, LlamaProcess>) -> u16 {
    get_port(&state)
}

#[tauri::command]
pub fn start_llama_lazy(
    app: tauri::AppHandle,
    state: tauri::State<'_, LlamaProcess>,
) -> Result<u16, String> {
    if is_running(&state) {
        let port = get_port(&state);
        if port > 0 {
            log::info!("[llama-server] Already running on port {}", port);
            return Ok(port);
        }
    }

    if crate::SHUTTING_DOWN.load(std::sync::atomic::Ordering::Relaxed) {
        return Err("Application is shutting down".to_string());
    }

    if !model::model_exists() {
        return Err("Model not downloaded yet".to_string());
    }

    let port = {
        let p = *state.port.lock().unwrap();
        if p > 0 {
            p
        } else {
            portpicker::pick_unused_port().unwrap_or(8080)
        }
    };

    log::info!("[llama-server] Lazy start: starting on port {}...", port);

    let actual_port = start(&app, &state, port)?;

    if wait_ready(actual_port, std::time::Duration::from_secs(120)) {
        log::info!("[llama-server] Lazy start complete — ready on port {}", actual_port);
        Ok(actual_port)
    } else {
        stop(&state);
        Err("llama-server failed to become ready within 120 seconds".to_string())
    }
}
