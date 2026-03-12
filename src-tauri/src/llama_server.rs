use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use tauri::Manager;

use crate::model;

#[derive(Debug, Clone, Copy, PartialEq)]
enum GpuBackend {
    Metal,
    Cuda,
    Rocm,
    CpuOnly,
}

fn detect_gpu_backend() -> GpuBackend {
    if cfg!(target_os = "macos") {
        log::info!("[GPU] Detected macOS — using Metal / MLX backend");
        return GpuBackend::Metal;
    }

    if let Ok(output) = Command::new("nvidia-smi")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        if output.success() {
            log::info!("[GPU] Detected NVIDIA GPU — using CUDA acceleration");
            return GpuBackend::Cuda;
        }
    }

    if let Ok(output) = Command::new("rocminfo")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        if output.success() {
            log::info!("[GPU] Detected AMD GPU — using ROCm backend");
            return GpuBackend::Rocm;
        }
    }

    log::info!("[GPU] No GPU detected — using optimized CPU inference (AVX)");
    GpuBackend::CpuOnly
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
    let resolved = path.canonicalize().unwrap_or(path.to_path_buf());
    let mut result = resolved.to_string_lossy().to_string();
    // Strip Windows UNC \\?\ prefix — it can break DLL loading when used
    // with Command::new() because the DLL loader may not resolve the exe's
    // directory correctly from a \\?\ path.
    if result.starts_with(r"\\?\") {
        result = result[4..].to_string();
    }
    Some(result)
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

    // Collect all directories that may contain DLLs/dylibs
    let dll_dirs = find_lib_directories(app, binary_dir);

    // Best-effort: try to copy libs next to binary (may fail on Windows
    // if binary is in Program Files — that's OK, we use PATH fallback)
    for dir in &dll_dirs {
        copy_libs_from_dir(dir, binary_dir);
    }

    let gpu = detect_gpu_backend();
    let model_path_str = model_path.to_string_lossy().to_string();
    let port_str = port.to_string();
    let threads_str = threads.to_string();

    let mut args: Vec<String> = vec![
        "--model".into(), model_path_str,
        "--port".into(), port_str,
        "--host".into(), "127.0.0.1".into(),
        "--ctx-size".into(), "4096".into(),
        "--threads".into(), threads_str,
        "--cache-type-k".into(), "q8_0".into(),
        "--cache-type-v".into(), "q8_0".into(),
        "--cache-reuse".into(), "256".into(),
        "--defrag-thold".into(), "0.1".into(),
        "--slot-prompt-similarity".into(), "0.5".into(),
        "--mlock".into(),
        "--batch-size".into(), "512".into(),
        "--ubatch-size".into(), "256".into(),
    ];

    match gpu {
        GpuBackend::Metal => {
            log::info!("[llama-server] Configuring for Metal: full GPU offload + flash attention");
            args.extend(["--n-gpu-layers".into(), "99".into()]);
            args.push("--flash-attn".into());
        }
        GpuBackend::Cuda => {
            log::info!("[llama-server] Configuring for CUDA: full GPU offload + flash attention");
            args.extend(["--n-gpu-layers".into(), "99".into()]);
            args.push("--flash-attn".into());
        }
        GpuBackend::Rocm => {
            log::info!("[llama-server] Configuring for ROCm: full GPU offload");
            args.extend(["--n-gpu-layers".into(), "99".into()]);
        }
        GpuBackend::CpuOnly => {
            log::info!("[llama-server] Configuring for CPU-only: no GPU offload");
            args.extend(["--n-gpu-layers".into(), "0".into()]);
        }
    }

    log::info!("[llama-server] GPU backend: {:?}, args count: {}", gpu, args.len());

    let mut cmd = Command::new(&binary);
    cmd.args(&args)
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    // On Windows, add all DLL directories to PATH so the loader finds them.
    // This is the primary DLL resolution mechanism — copying is just a fallback.
    // Solves: "ggml-base.dll was not found" errors even when DLLs exist
    // in resource dirs that aren't on the default search path.
    #[cfg(target_os = "windows")]
    {
        let mut path_dirs: Vec<String> = Vec::new();
        path_dirs.push(binary_dir.to_string_lossy().to_string());
        for dir in &dll_dirs {
            path_dirs.push(dir.to_string_lossy().to_string());
        }
        let system_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{};{}", path_dirs.join(";"), system_path);
        cmd.env("PATH", new_path);
        log::info!("[llama-server] Windows DLL search paths: {:?}", path_dirs);
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
        if let Ok(meta) = std::fs::metadata(&binary) {
            let mode = meta.permissions().mode();
            if mode & 0o111 == 0 {
                log::warn!("[llama-server] Binary lacks execute permission, fixing...");
                let mut perms = meta.permissions();
                perms.set_mode(mode | 0o755);
                let _ = std::fs::set_permissions(&binary, perms);
            }
        }
    }

    let child = cmd
        .spawn()
        .map_err(|e| {
            log::error!("[llama-server] Failed to spawn process: {}", e);
            format!("Failed to start llama-server: {}", e)
        })?;

    log::info!("[llama-server] Process spawned (PID: {})", child.id());

    *state.child.lock().unwrap() = Some(child);
    *state.port.lock().unwrap() = port;

    Ok(port)
}

pub fn wait_ready(port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    let url = format!("http://127.0.0.1:{}/health", port);
    log::info!("[llama-server] Waiting for health at {} (timeout: {:?})", url, timeout);

    let mut attempt = 0;
    while start.elapsed() < timeout {
        attempt += 1;
        match reqwest::blocking::get(&url) {
            Ok(resp) if resp.status().is_success() => {
                log::info!(
                    "[llama-server] Ready on port {} (took {:.1}s, {} attempts)",
                    port,
                    start.elapsed().as_secs_f64(),
                    attempt
                );
                return true;
            }
            Ok(resp) => {
                if attempt <= 3 || attempt % 10 == 0 {
                    log::info!("[llama-server] Health check attempt {}: status {}", attempt, resp.status());
                }
            }
            Err(e) => {
                if attempt <= 3 || attempt % 10 == 0 {
                    log::info!("[llama-server] Health check attempt {}: {}", attempt, e);
                }
            }
        }
        thread::sleep(Duration::from_millis(1000));
    }

    log::error!(
        "[llama-server] Failed to become ready within {:?} ({} attempts)",
        timeout,
        attempt
    );
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
