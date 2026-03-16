use std::io::BufRead;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use tauri::Manager;

mod model;
mod llama_server;

static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);
static NODE_PORT: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(0);

// ─── Windows Job Object ──────────────────────────────────────────────────────
// Creates a Win32 Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
// When the Epito process exits (even if it crashes), Windows automatically
// terminates ALL child processes in the job. This is how Chrome, VSCode,
// and other enterprise apps guarantee zero orphan processes.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(windows)]
mod win_job {
    use std::sync::OnceLock;

    static JOB: OnceLock<JobHandle> = OnceLock::new();

    struct JobHandle(isize);
    unsafe impl Send for JobHandle {}
    unsafe impl Sync for JobHandle {}

    impl Drop for JobHandle {
        fn drop(&mut self) {
            if self.0 != 0 {
                unsafe { CloseHandle(self.0); }
            }
        }
    }

    extern "system" {
        fn CreateJobObjectW(attrs: *const u8, name: *const u16) -> isize;
        fn SetInformationJobObject(job: isize, class: u32, info: *const u8, len: u32) -> i32;
        fn AssignProcessToJobObject(job: isize, process: isize) -> i32;
        fn CloseHandle(handle: isize) -> i32;
    }

    // JOBOBJECT_BASIC_LIMIT_INFORMATION for x86_64
    #[repr(C)]
    struct BasicLimitInfo {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    struct IoCounters {
        read_ops: u64, write_ops: u64, other_ops: u64,
        read_bytes: u64, write_bytes: u64, other_bytes: u64,
    }

    #[repr(C)]
    struct ExtendedLimitInfo {
        basic: BasicLimitInfo,
        io: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory: usize,
        peak_job_memory: usize,
    }

    pub fn init() {
        JOB.get_or_init(|| unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job == 0 {
                log::warn!("[Windows] Failed to create Job Object");
                return JobHandle(0);
            }

            let mut info: ExtendedLimitInfo = std::mem::zeroed();
            info.basic.limit_flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            let ok = SetInformationJobObject(
                job,
                9, // JobObjectExtendedLimitInformation
                &info as *const _ as *const u8,
                std::mem::size_of::<ExtendedLimitInfo>() as u32,
            );

            if ok == 0 {
                log::warn!("[Windows] Failed to configure Job Object");
            } else {
                log::info!("[Windows] Job Object active — all child processes will auto-terminate on exit");
            }

            JobHandle(job)
        });
    }

    pub fn assign(child: &std::process::Child) {
        if let Some(job) = JOB.get() {
            if job.0 == 0 { return; }
            unsafe {
                use std::os::windows::io::AsRawHandle;
                let ok = AssignProcessToJobObject(job.0, child.as_raw_handle() as isize);
                if ok == 0 {
                    log::warn!("[Windows] Failed to assign PID {} to Job Object", child.id());
                } else {
                    log::info!("[Windows] PID {} assigned to Job Object", child.id());
                }
            }
        }
    }
}

struct ServerProcess(Mutex<Option<Child>>);

impl Drop for ServerProcess {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.0.lock() {
            if let Some(ref mut child) = *guard {
                kill_process_tree(child);
            }
            *guard = None;
        }
    }
}

fn find_free_port() -> u16 {
    portpicker::pick_unused_port().unwrap_or(3210)
}

fn wait_for_server(port: u16, timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    let url = format!("http://127.0.0.1:{}/api/ping", port);

    while start.elapsed() < timeout {
        if SHUTTING_DOWN.load(Ordering::Relaxed) {
            return false;
        }
        match reqwest::blocking::get(&url) {
            Ok(resp) if resp.status().is_success() => return true,
            _ => thread::sleep(Duration::from_millis(500)),
        }
    }
    false
}

/// Search a directory for a binary by base name.
/// Checks both plain name ("node.exe") and Tauri sidecar pattern
/// ("node-x86_64-pc-windows-msvc.exe").
pub(crate) fn find_binary_in_dir(dir: &std::path::Path, base_name: &str) -> Option<std::path::PathBuf> {
    if !dir.exists() {
        return None;
    }
    let ext = if cfg!(target_os = "windows") { ".exe" } else { "" };

    let exact = dir.join(format!("{}{}", base_name, ext));
    if exact.exists() {
        return Some(exact);
    }

    let prefix = format!("{}-", base_name);
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && name.ends_with(ext) && !name.contains("tmp") {
                return Some(entry.path());
            }
        }
    }

    None
}

fn find_node(app: &tauri::AppHandle) -> Option<String> {
    if let Ok(resource_dir) = app.path().resource_dir() {
        log::info!("[Node] Checking resource dir: {:?}", resource_dir);
        if let Some(p) = find_binary_in_dir(&resource_dir, "node") {
            if let Some(path) = check_node_binary(&p) {
                return Some(path);
            }
        }
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            log::info!("[Node] Checking exe dir: {:?}", exe_dir);
            if let Some(p) = find_binary_in_dir(exe_dir, "node") {
                if let Some(path) = check_node_binary(&p) {
                    return Some(path);
                }
            }
        }
    }

    for dev_dir in &["src-tauri/binaries", "binaries"] {
        let dir = std::path::PathBuf::from(dev_dir);
        log::info!("[Node] Checking dev path: {:?}", dir);
        if let Some(p) = find_binary_in_dir(&dir, "node") {
            if let Some(path) = check_node_binary(&p) {
                return Some(path);
            }
        }
    }

    let candidates = if cfg!(target_os = "windows") {
        vec!["node.exe".to_string(), r"C:\Program Files\nodejs\node.exe".to_string()]
    } else {
        vec!["node".to_string(), "/usr/local/bin/node".to_string(),
             "/opt/homebrew/bin/node".to_string(), "/usr/bin/node".to_string()]
    };

    for candidate in candidates {
        if Command::new(&candidate).arg("--version").output().is_ok() {
            log::warn!("[Node] Using system Node.js: {} (ABI mismatch risk!)", candidate);
            return Some(candidate);
        }
    }

    log::error!("[Node] Node.js binary not found anywhere");
    None
}

fn check_node_binary(path: &std::path::Path) -> Option<String> {
    if !path.exists() { return None; }
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if size < 1_000_000 {
        log::warn!("[Node] Binary too small ({} bytes): {:?}", size, path);
        return None;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            let mode = meta.permissions().mode();
            if mode & 0o111 == 0 {
                let mut perms = meta.permissions();
                perms.set_mode(mode | 0o755);
                let _ = std::fs::set_permissions(path, perms);
            }
        }
    }

    log::info!("[Node] Found: {:?} ({:.1} MB)", path, size as f64 / 1_048_576.0);
    Some(normalize_windows_path(path).to_string_lossy().to_string())
}

pub(crate) fn normalize_windows_path(path: &std::path::Path) -> std::path::PathBuf {
    let resolved = path.canonicalize().unwrap_or(path.to_path_buf());
    #[cfg(windows)]
    {
        let s = resolved.to_string_lossy();
        if let Some(stripped) = s.strip_prefix(r"\\?\") {
            return std::path::PathBuf::from(stripped);
        }
    }
    resolved
}

pub(crate) fn kill_process_tree(child: &mut Child) {
    let pid = child.id();

    #[cfg(unix)]
    {
        let _ = Command::new("kill").args(["-15", &format!("-{}", pid)])
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
        thread::sleep(Duration::from_millis(500));
        let _ = Command::new("kill").args(["-9", &format!("-{}", pid)])
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .creation_flags(0x08000000)
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
            .status();
    }

    let _ = child.kill();
    let _ = child.wait();
}

// ─── Window State ────────────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct WindowState {
    width: u32, height: u32, x: i32, y: i32, maximized: bool,
}

fn window_state_path() -> std::path::PathBuf {
    dirs::home_dir().expect("Cannot determine home directory")
        .join(".epito").join("window-state.json")
}

fn load_window_state() -> Option<WindowState> {
    let data = std::fs::read_to_string(window_state_path()).ok()?;
    let state: WindowState = serde_json::from_str(&data).ok()?;
    if state.width < 400 || state.height < 300 || state.width > 10000 || state.height > 10000 {
        return None;
    }
    Some(state)
}

fn save_window_state(state: &WindowState) {
    let path = window_state_path();
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent).ok(); }
    if let Ok(json) = serde_json::to_string_pretty(state) { std::fs::write(&path, json).ok(); }
}

fn save_current_window_state(window: &tauri::WebviewWindow) {
    let size = match window.outer_size() { Ok(s) => s, Err(_) => return };
    let pos = match window.outer_position() { Ok(p) => p, Err(_) => return };
    let maximized = window.is_maximized().unwrap_or(false);
    save_window_state(&WindowState {
        width: size.width, height: size.height, x: pos.x, y: pos.y, maximized,
    });
}

fn restore_window_state(window: &tauri::WebviewWindow) {
    let state = match load_window_state() {
        Some(s) => s,
        None => return,
    };
    if state.maximized {
        let _ = window.maximize();
        return;
    }
    let _ = window.set_size(tauri::PhysicalSize::new(state.width, state.height));
    let (x, y) = validate_position(window, &state);
    let _ = window.set_position(tauri::PhysicalPosition::new(x, y));
}

fn validate_position(window: &tauri::WebviewWindow, state: &WindowState) -> (i32, i32) {
    if let Ok(monitors) = window.available_monitors() {
        for m in &monitors {
            let mp = m.position();
            let ms = m.size();
            if state.x >= mp.x - (state.width as i32 - 100)
                && state.x < mp.x + ms.width as i32
                && state.y >= mp.y
                && state.y < mp.y + ms.height as i32
            {
                return (state.x, state.y);
            }
        }
        if let Ok(Some(primary)) = window.primary_monitor() {
            let mp = primary.position();
            let ms = primary.size();
            return (
                (mp.x + (ms.width as i32 - state.width as i32) / 2).max(0),
                (mp.y + (ms.height as i32 - state.height as i32) / 2).max(0),
            );
        }
    }
    (state.x, state.y)
}

// ─── File Dialog ─────────────────────────────────────────────────────────────

#[tauri::command]
async fn save_file_with_dialog(
    app: tauri::AppHandle, data: Vec<u8>, default_name: String,
    filter_name: String, filter_extensions: Vec<String>,
) -> Result<bool, String> {
    use tauri_plugin_dialog::DialogExt;
    let ext_refs: Vec<&str> = filter_extensions.iter().map(|s| s.as_str()).collect();
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().set_file_name(&default_name)
        .add_filter(&filter_name, &ext_refs)
        .save_file(move |path| { let _ = tx.send(path); });
    let file_path = rx.await.map_err(|_| "Dialog cancelled".to_string())?;
    match file_path {
        Some(path) => {
            let actual = path.as_path().ok_or("Invalid path")?;
            tokio::fs::write(actual, &data).await.map_err(|e| format!("Write failed: {}", e))?;
            Ok(true)
        }
        None => Ok(false),
    }
}

// ─── Shutdown ────────────────────────────────────────────────────────────────

fn perform_shutdown(app_handle: &tauri::AppHandle) {
    if SHUTTING_DOWN.swap(true, Ordering::SeqCst) { return; }

    log::info!("[Lifecycle] === SHUTDOWN SEQUENCE STARTED ===");

    if let Some(window) = app_handle.get_webview_window("main") {
        save_current_window_state(&window);
    }

    // Graceful llama-server unload
    let llama_state = app_handle.state::<llama_server::LlamaProcess>();
    let llama_port = llama_server::get_port(&llama_state);
    if llama_port > 0 {
        let _ = reqwest::blocking::Client::new()
            .post(format!("http://127.0.0.1:{}/slots/0?action=erase", llama_port))
            .timeout(Duration::from_secs(2)).send();
    }

    // Graceful Node.js shutdown (closes SQLite)
    let np = NODE_PORT.load(Ordering::Relaxed);
    if np > 0 {
        let _ = reqwest::blocking::Client::new()
            .post(format!("http://127.0.0.1:{}/api/shutdown", np))
            .timeout(Duration::from_secs(2)).send();
        thread::sleep(Duration::from_millis(500));
    }

    // Force-kill Node.js
    let server_state = app_handle.state::<ServerProcess>();
    if let Ok(mut guard) = server_state.0.lock() {
        if let Some(ref mut child) = *guard {
            log::info!("[Lifecycle] Killing Node.js (PID: {})", child.id());
            kill_process_tree(child);
        }
        *guard = None;
    }

    // Force-kill llama-server
    llama_server::stop(&llama_state);

    // Final sweep: kill ANY orphaned processes by all known names.
    // This catches processes that escaped PID-based killing.
    sweep_orphan_processes();

    log::info!("[Lifecycle] === SHUTDOWN COMPLETE ===");
}

fn sweep_orphan_processes() {
    #[cfg(unix)]
    {
        let _ = Command::new("pkill").args(["-f", "llama-server.*--host.*127.0.0.1"])
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // Kill by ALL possible process names (exact name AND sidecar name)
        for name in &[
            "llama-server.exe",
            "llama-server-x86_64-pc-windows-msvc.exe",
            "node-x86_64-pc-windows-msvc.exe",
        ] {
            let _ = Command::new("taskkill")
                .args(["/F", "/IM", name])
                .creation_flags(0x08000000)
                .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
                .status();
        }
    }
}

fn install_signal_handlers(app_handle: tauri::AppHandle) {
    #[cfg(unix)]
    {
        let handle = app_handle.clone();
        thread::spawn(move || {
            let (tx, rx) = std::sync::mpsc::channel();
            let tx_int = tx.clone();
            let _ = ctrlc_handler(move || { let _ = tx_int.send("SIGINT"); });
            if let Ok(signal) = rx.recv() {
                log::info!("[Lifecycle] {} — shutting down", signal);
                perform_shutdown(&handle);
                std::process::exit(0);
            }
        });
    }

    #[cfg(windows)]
    {
        let handle = app_handle.clone();
        let _ = ctrlc_handler(move || {
            perform_shutdown(&handle);
            std::process::exit(0);
        });
    }
}

fn ctrlc_handler<F: FnOnce() + Send + 'static>(handler: F) -> Result<(), String> {
    let once = std::sync::Once::new();
    let handler = std::sync::Mutex::new(Some(handler));
    ctrlc::set_handler(move || {
        once.call_once(|| {
            if let Ok(mut g) = handler.lock() { if let Some(h) = g.take() { h(); } }
        });
    }).map_err(|e| format!("{}", e))
}

// ─── App Entry ───────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let node_port = find_free_port();
    NODE_PORT.store(node_port, Ordering::Relaxed);

    // Windows: Create Job Object BEFORE spawning any children.
    // All children assigned to this job auto-terminate when Epito exits.
    #[cfg(windows)]
    win_job::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(ServerProcess(Mutex::new(None)))
        .manage(llama_server::LlamaProcess::new())
        .invoke_handler(tauri::generate_handler![
            model::check_model,
            model::download_model,
            model::delete_model,
            model::get_download_progress,
            llama_server::get_llama_port,
            llama_server::start_llama_lazy,
            llama_server::stop_llama_idle,
            save_file_with_dialog,
        ])
        .setup(move |app| {
            app.handle().plugin(
                tauri_plugin_log::Builder::default().level(log::LevelFilter::Info).build(),
            )?;

            log::info!("[Lifecycle] Application starting");
            install_signal_handlers(app.handle().clone());

            if let Some(window) = app.get_webview_window("main") {
                restore_window_state(&window);
                // Set dark background on about:blank immediately to prevent white flash.
                // The React BrandedSplash component handles the actual animation.
                let _ = window.eval("document.documentElement.style.background='#0a0a0f';document.body.style.background='#0a0a0f';document.body.style.margin='0';");
            }

            if cfg!(debug_assertions) {
                let window = app.get_webview_window("main").unwrap();
                window.navigate("http://127.0.0.1:3000".parse().unwrap())?;
                let handle = app.handle().clone();
                thread::spawn(move || {
                    let llama_state = handle.state::<llama_server::LlamaProcess>();
                    if model::model_exists() {
                        let port = portpicker::pick_unused_port().unwrap_or(8080);
                        if let Ok(p) = llama_server::start(&handle, &llama_state, port) {
                            llama_server::wait_ready(p, Duration::from_secs(120));
                        }
                    }
                });
                return Ok(());
            }

            // ── Production startup ──

            let node_path = find_node(app.handle())
                .ok_or("Node.js runtime not found")?;

            let resource_dir = app.path().resource_dir()
                .map_err(|e| format!("Resource dir error: {}", e))?;

            let standalone_dir = normalize_windows_path(
                &resource_dir.join("_up_").join(".next").join("standalone"));
            let server_js = normalize_windows_path(&standalone_dir.join("server.js"));
            log::info!("[Node] Standalone: {:?}", standalone_dir);
            log::info!("[Node] Entry: {:?}", server_js);

            if !server_js.exists() {
                return Err(format!("server.js not found at {:?}", server_js).into());
            }

            let data_dir = model::data_dir();
            std::fs::create_dir_all(&data_dir).ok();

            let llama_port = find_free_port();
            let data_dir_str = data_dir.to_string_lossy().to_string();
            let node_path_clone = node_path.clone();
            let standalone_dir_clone = standalone_dir.clone();
            let server_js_clone = server_js.clone();

            log::info!("[Lifecycle] node_port={}, llama_port={}", node_port, llama_port);

            {
                let llama_state = app.state::<llama_server::LlamaProcess>();
                *llama_state.port.lock().unwrap() = llama_port;
            }

            // ── Spawn Node.js server ──
            let handle_node = app.handle().clone();
            thread::spawn(move || {
                if SHUTTING_DOWN.load(Ordering::Relaxed) { return; }

                let mut cmd = Command::new(&node_path_clone);
                cmd.args([server_js_clone.to_string_lossy().to_string()])
                    .current_dir(&standalone_dir_clone)
                    .env("PORT", node_port.to_string())
                    .env("HOSTNAME", "127.0.0.1")
                    .env("NODE_ENV", "production")
                    .env("EPITO_DATA_DIR", &data_dir_str)
                    .env("LLAMA_SERVER_PORT", llama_port.to_string())
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped());

                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
                }

                #[cfg(unix)]
                {
                    use std::os::unix::process::CommandExt;
                    cmd.process_group(0);
                }

                match cmd.spawn() {
                    Ok(mut child) => {
                        // Assign to Job Object — auto-kills when Epito exits
                        #[cfg(windows)]
                        win_job::assign(&child);

                        if let Some(stderr) = child.stderr.take() {
                            thread::spawn(move || {
                                let reader = std::io::BufReader::new(stderr);
                                for line in reader.lines().flatten() {
                                    if !line.is_empty() { log::warn!("[Node.js] {}", line); }
                                }
                            });
                        }

                        let server_state = handle_node.state::<ServerProcess>();
                        *server_state.0.lock().unwrap() = Some(child);

                        let url = format!("http://127.0.0.1:{}", node_port);
                        if wait_for_server(node_port, Duration::from_secs(30)) {
                            log::info!("[Lifecycle] Node.js ready on port {}", node_port);
                        } else {
                            log::error!("[Lifecycle] Node.js timeout — navigating anyway");
                        }
                        if let Some(window) = handle_node.get_webview_window("main") {
                            let _ = window.navigate(url.parse().unwrap());
                        }
                    }
                    Err(e) => log::error!("[Lifecycle] Node.js spawn failed: {}", e),
                }
            });

            // ── Spawn llama-server + idle lifecycle manager ──
            let handle_llama = app.handle().clone();
            let idle_data_dir = data_dir.clone();
            thread::spawn(move || {
                if model::model_exists() {
                    let llama_state = handle_llama.state::<llama_server::LlamaProcess>();
                    match llama_server::start(&handle_llama, &llama_state, llama_port) {
                        Ok(port) => {
                            if !llama_server::wait_ready(port, Duration::from_secs(120)) {
                                log::error!("[Lifecycle] llama-server timeout (120s)");
                            }
                        }
                        Err(e) => log::warn!("[Lifecycle] llama-server: {}", e),
                    }
                } else {
                    log::info!("[Lifecycle] No model — will prompt download");
                }

                // Idle lifecycle: watch for signal files from Node.js
                // .idle-stop  → kill llama-server to free memory
                // .idle-start → restart llama-server for next AI request
                let stop_signal = idle_data_dir.join(".idle-stop");
                let start_signal = idle_data_dir.join(".idle-start");
                loop {
                    if SHUTTING_DOWN.load(Ordering::Relaxed) { break; }
                    thread::sleep(Duration::from_secs(2));

                    if stop_signal.exists() {
                        let _ = std::fs::remove_file(&stop_signal);
                        let llama_state = handle_llama.state::<llama_server::LlamaProcess>();
                        if llama_server::is_running(&llama_state) {
                            log::info!("[Lifecycle] Idle stop signal — killing llama-server to free memory");
                            llama_server::stop(&llama_state);
                        }
                    }

                    if start_signal.exists() {
                        let _ = std::fs::remove_file(&start_signal);
                        let llama_state = handle_llama.state::<llama_server::LlamaProcess>();
                        if !llama_server::is_running(&llama_state) && model::model_exists() {
                            log::info!("[Lifecycle] Idle start signal — restarting llama-server");
                            match llama_server::start(&handle_llama, &llama_state, llama_port) {
                                Ok(port) => {
                                    llama_server::wait_ready(port, Duration::from_secs(120));
                                }
                                Err(e) => log::warn!("[Lifecycle] Restart failed: {}", e),
                            }
                        }
                    }
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Epito")
        .run(|app_handle, event| {
            match event {
                tauri::RunEvent::WindowEvent {
                    label, event: tauri::WindowEvent::CloseRequested { .. }, ..
                } => {
                    if label == "main" {
                        perform_shutdown(app_handle);
                    }
                }
                tauri::RunEvent::Exit => {
                    perform_shutdown(app_handle);
                }
                _ => {}
            }
        });
}
