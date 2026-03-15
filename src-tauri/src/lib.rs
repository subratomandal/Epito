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
    // Use /api/ping — a zero-dependency endpoint that imports nothing.
    // Unlike /api/health, it doesn't import better-sqlite3 or any native
    // module, so it works even if native modules fail to load on Windows.
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
/// Used by both find_node and llama_server::find_llama_server.
/// Checks both the plain name (e.g. "node.exe") and the Tauri sidecar
/// name with target triple (e.g. "node-x86_64-pc-windows-msvc.exe").
/// The sidecar name MUST be checked because Tauri's externalBin bundles
/// binaries with the triple suffix.
pub(crate) fn find_binary_in_dir(dir: &std::path::Path, base_name: &str) -> Option<std::path::PathBuf> {
    if !dir.exists() {
        return None;
    }
    let ext = if cfg!(target_os = "windows") { ".exe" } else { "" };

    // Try exact name first: "node.exe"
    let exact = dir.join(format!("{}{}", base_name, ext));
    if exact.exists() {
        return Some(exact);
    }

    // Try Tauri sidecar pattern: "node-{triple}.exe"
    // Scan directory for any file starting with "{base_name}-" and ending with the ext
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
    // 1. Check Tauri resource directory (bundled binary in production)
    if let Ok(resource_dir) = app.path().resource_dir() {
        log::info!("[Node] Checking resource dir: {:?}", resource_dir);
        if let Some(p) = find_binary_in_dir(&resource_dir, "node") {
            if let Some(path) = check_node_binary(&p) {
                return Some(path);
            }
        }
    }

    // 2. Check next to the main executable (sidecar location — where Tauri places externalBin)
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

    // 3. Check dev paths
    for dev_dir in &["src-tauri/binaries", "binaries"] {
        let dir = std::path::PathBuf::from(dev_dir);
        log::info!("[Node] Checking dev path: {:?}", dir);
        if let Some(p) = find_binary_in_dir(&dir, "node") {
            if let Some(path) = check_node_binary(&p) {
                return Some(path);
            }
        }
    }

    // 4. Fall back to system Node.js (last resort — version may not match native modules!)
    let candidates = if cfg!(target_os = "windows") {
        vec![
            "node.exe".to_string(),
            r"C:\Program Files\nodejs\node.exe".to_string(),
        ]
    } else {
        vec![
            "node".to_string(),
            "/usr/local/bin/node".to_string(),
            "/opt/homebrew/bin/node".to_string(),
            "/usr/bin/node".to_string(),
        ]
    };

    for candidate in candidates {
        let result = Command::new(&candidate).arg("--version").output();
        if result.is_ok() {
            log::warn!("[Node] Using system Node.js: {} (native module ABI mismatch possible!)", candidate);
            return Some(candidate);
        }
    }

    log::error!("[Node] Node.js binary not found anywhere");
    None
}

fn check_node_binary(path: &std::path::Path) -> Option<String> {
    if !path.exists() {
        return None;
    }
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if size < 1_000_000 {
        log::warn!("[Node] Binary at {:?} is too small ({} bytes), skipping", path, size);
        return None;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            let mode = meta.permissions().mode();
            if mode & 0o111 == 0 {
                log::warn!("[Node] Binary lacks execute permission, fixing...");
                let mut perms = meta.permissions();
                perms.set_mode(mode | 0o755);
                let _ = std::fs::set_permissions(path, perms);
            }
        }
    }

    log::info!("[Node] Found valid binary at {:?} ({:.1} MB)", path, size as f64 / 1_048_576.0);
    Some(normalize_windows_path(path).to_string_lossy().to_string())
}

/// Resolve a path and strip the Windows UNC `\\?\` prefix.
/// UNC paths break DLL loading and some Node.js module resolution.
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
        let _ = Command::new("kill")
            .args(["-15", &format!("-{}", pid)])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        thread::sleep(Duration::from_millis(500));

        let _ = Command::new("kill")
            .args(["-9", &format!("-{}", pid)])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .creation_flags(0x08000000)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    let _ = child.kill();
    let _ = child.wait();
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct WindowState {
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    maximized: bool,
}

fn window_state_path() -> std::path::PathBuf {
    dirs::home_dir()
        .expect("Cannot determine home directory")
        .join(".epito")
        .join("window-state.json")
}

fn load_window_state() -> Option<WindowState> {
    let path = window_state_path();
    let data = std::fs::read_to_string(&path).ok()?;
    let state: WindowState = serde_json::from_str(&data).ok()?;
    if state.width < 400 || state.height < 300 || state.width > 10000 || state.height > 10000 {
        return None;
    }
    Some(state)
}

fn save_window_state(state: &WindowState) {
    let path = window_state_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(json) = serde_json::to_string_pretty(state) {
        std::fs::write(&path, json).ok();
    }
}

fn save_current_window_state(window: &tauri::WebviewWindow) {
    let size = match window.outer_size() {
        Ok(s) => s,
        Err(_) => return,
    };
    let pos = match window.outer_position() {
        Ok(p) => p,
        Err(_) => return,
    };
    let maximized = window.is_maximized().unwrap_or(false);

    let state = WindowState {
        width: size.width,
        height: size.height,
        x: pos.x,
        y: pos.y,
        maximized,
    };

    log::info!(
        "[Window] Saving state: {}x{} at ({},{}) maximized={}",
        state.width, state.height, state.x, state.y, state.maximized
    );
    save_window_state(&state);
}

fn restore_window_state(window: &tauri::WebviewWindow) {
    let state = match load_window_state() {
        Some(s) => s,
        None => {
            log::info!("[Window] No saved state found — using defaults");
            return;
        }
    };

    log::info!(
        "[Window] Restoring state: {}x{} at ({},{}) maximized={}",
        state.width, state.height, state.x, state.y, state.maximized
    );

    if state.maximized {
        let _ = window.maximize();
        return;
    }

    let _ = window.set_size(tauri::PhysicalSize::new(state.width, state.height));

    let (safe_x, safe_y) = validate_position(window, &state);
    let _ = window.set_position(tauri::PhysicalPosition::new(safe_x, safe_y));
}

fn validate_position(window: &tauri::WebviewWindow, state: &WindowState) -> (i32, i32) {
    if let Ok(monitors) = window.available_monitors() {
        for monitor in &monitors {
            let mpos = monitor.position();
            let msize = monitor.size();
            let right = mpos.x + msize.width as i32;
            let bottom = mpos.y + msize.height as i32;

            if state.x >= mpos.x - (state.width as i32 - 100)
                && state.x < right
                && state.y >= mpos.y
                && state.y < bottom
            {
                return (state.x, state.y);
            }
        }

        if let Ok(Some(primary)) = window.primary_monitor() {
            let mpos = primary.position();
            let msize = primary.size();
            let x = mpos.x + (msize.width as i32 - state.width as i32) / 2;
            let y = mpos.y + (msize.height as i32 - state.height as i32) / 2;
            log::info!("[Window] Saved position off-screen — recentering to ({}, {})", x, y);
            return (x.max(0), y.max(0));
        }
    }

    (state.x, state.y)
}

#[tauri::command]
async fn save_file_with_dialog(
    app: tauri::AppHandle,
    data: Vec<u8>,
    default_name: String,
    filter_name: String,
    filter_extensions: Vec<String>,
) -> Result<bool, String> {
    use tauri_plugin_dialog::DialogExt;
    use tokio::sync::oneshot;

    let ext_refs: Vec<&str> = filter_extensions.iter().map(|s| s.as_str()).collect();

    let (tx, rx) = oneshot::channel();

    app.dialog()
        .file()
        .set_file_name(&default_name)
        .add_filter(&filter_name, &ext_refs)
        .save_file(move |path| {
            let _ = tx.send(path);
        });

    let file_path = rx.await.map_err(|_| "Dialog cancelled".to_string())?;

    match file_path {
        Some(path) => {
            let actual_path = path.as_path()
                .ok_or_else(|| "Invalid file path from dialog".to_string())?;
            tokio::fs::write(actual_path, &data)
                .await
                .map_err(|e| format!("Failed to write file: {}", e))?;
            log::info!("[Export] File saved to {:?}", actual_path);
            Ok(true)
        }
        None => {
            log::info!("[Export] Save cancelled by user");
            Ok(false)
        }
    }
}

fn perform_shutdown(app_handle: &tauri::AppHandle) {
    if SHUTTING_DOWN.swap(true, Ordering::SeqCst) {
        return;
    }

    log::info!("[Lifecycle] === SHUTDOWN SEQUENCE STARTED ===");

    if let Some(window) = app_handle.get_webview_window("main") {
        save_current_window_state(&window);
    }

    let llama_state = app_handle.state::<llama_server::LlamaProcess>();
    let llama_port = llama_server::get_port(&llama_state);
    if llama_port > 0 {
        log::info!("[Lifecycle] Requesting llama-server model unload...");
        let _ = reqwest::blocking::Client::new()
            .post(format!("http://127.0.0.1:{}/slots/0?action=erase", llama_port))
            .timeout(Duration::from_secs(2))
            .send();
    }

    // Request graceful Node.js shutdown via HTTP before force-killing.
    // This gives the Node.js process a chance to close the SQLite database
    // cleanly (flush WAL, release locks). Critical on Windows where
    // taskkill /F gives no opportunity for cleanup code to run.
    let np = NODE_PORT.load(Ordering::Relaxed);
    if np > 0 {
        log::info!("[Lifecycle] Requesting Node.js graceful shutdown on port {}...", np);
        let _ = reqwest::blocking::Client::new()
            .post(format!("http://127.0.0.1:{}/api/shutdown", np))
            .timeout(Duration::from_secs(2))
            .send();
        thread::sleep(Duration::from_millis(500));
    }

    let server_state = app_handle.state::<ServerProcess>();
    if let Ok(mut guard) = server_state.0.lock() {
        if let Some(ref mut child) = *guard {
            log::info!("[Lifecycle] Killing Node.js process tree (PID: {})...", child.id());
            kill_process_tree(child);
        }
        *guard = None;
    }

    log::info!("[Lifecycle] Killing llama-server...");
    llama_server::stop(&llama_state);

    sweep_orphan_processes();

    log::info!("[Lifecycle] === SHUTDOWN COMPLETE ===");
}

fn sweep_orphan_processes() {
    #[cfg(unix)]
    {
        let _ = Command::new("pkill")
            .args(["-f", "llama-server.*--host.*127.0.0.1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill")
            .args(["/F", "/IM", "llama-server.exe"])
            .creation_flags(0x08000000)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

fn install_signal_handlers(app_handle: tauri::AppHandle) {
    #[cfg(unix)]
    {
        let handle = app_handle.clone();
        thread::spawn(move || {
            use std::sync::mpsc;
            let (tx, rx) = mpsc::channel();
            let tx_int = tx.clone();
            let _ = ctrlc_handler(move || {
                let _ = tx_int.send("SIGINT");
            });
            if let Ok(signal) = rx.recv() {
                log::info!("[Lifecycle] Received {} — initiating shutdown...", signal);
                perform_shutdown(&handle);
                std::process::exit(0);
            }
        });
    }

    #[cfg(windows)]
    {
        let handle = app_handle.clone();
        let _ = ctrlc_handler(move || {
            log::info!("[Lifecycle] Received Ctrl+C — initiating shutdown...");
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
            if let Ok(mut guard) = handler.lock() {
                if let Some(h) = guard.take() {
                    h();
                }
            }
        });
    })
    .map_err(|e| format!("Failed to set Ctrl+C handler: {}", e))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let node_port = find_free_port();
    NODE_PORT.store(node_port, Ordering::Relaxed);

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
            save_file_with_dialog,
        ])
        .setup(move |app| {
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .build(),
            )?;

            log::info!("[Lifecycle] Application starting");

            install_signal_handlers(app.handle().clone());

            if let Some(window) = app.get_webview_window("main") {
                restore_window_state(&window);
            }

            if cfg!(debug_assertions) {
                let window = app.get_webview_window("main").unwrap();
                window.navigate("http://127.0.0.1:3000".parse().unwrap())?;

                let handle = app.handle().clone();
                thread::spawn(move || {
                    let llama_state = handle.state::<llama_server::LlamaProcess>();
                    if model::model_exists() {
                        let dev_port = portpicker::pick_unused_port().unwrap_or(8080);
                        match llama_server::start(&handle, &llama_state, dev_port) {
                            Ok(port) => {
                                if llama_server::wait_ready(port, Duration::from_secs(120)) {
                                    log::info!("[Lifecycle] llama-server ready on port {} (dev mode)", port);
                                }
                            }
                            Err(e) => log::warn!("[Lifecycle] Could not start llama-server in dev: {}", e),
                        }
                    }
                });

                return Ok(());
            }

            let node_path = match find_node(app.handle()) {
                Some(path) => path,
                None => {
                    log::error!("Node.js runtime not found");
                    return Err("Node.js runtime not found. The application may not have been built correctly.".into());
                }
            };

            let resource_dir = app
                .path()
                .resource_dir()
                .map_err(|e| format!("Failed to get resource directory: {}", e))?;

            let standalone_dir = normalize_windows_path(&resource_dir.join("_up_").join(".next").join("standalone"));
            let server_js = normalize_windows_path(&standalone_dir.join("server.js"));
            log::info!("[Node] Standalone dir: {:?}", standalone_dir);
            log::info!("[Node] Server entry: {:?}", server_js);

            if !server_js.exists() {
                log::error!("Standalone server not found at {:?}", server_js);
                return Err(format!(
                    "Standalone server not found at {:?}. The app may not have been built correctly.",
                    server_js
                )
                .into());
            }

            let data_dir = model::data_dir();
            std::fs::create_dir_all(&data_dir).ok();

            let llama_port = find_free_port();
            let data_dir_str = data_dir.to_string_lossy().to_string();
            let node_path_clone = node_path.clone();
            let standalone_dir_clone = standalone_dir.clone();
            let server_js_clone = server_js.clone();

            log::info!(
                "[Lifecycle] Production startup: node_port={}, llama_port={}",
                node_port, llama_port
            );

            {
                let llama_state = app.state::<llama_server::LlamaProcess>();
                *llama_state.port.lock().unwrap() = llama_port;
            }

            let handle_node = app.handle().clone();
            thread::spawn(move || {
                if SHUTTING_DOWN.load(Ordering::Relaxed) {
                    return;
                }

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

                match cmd.spawn() {
                    Ok(mut child) => {
                        if let Some(stderr) = child.stderr.take() {
                            thread::spawn(move || {
                                let reader = std::io::BufReader::new(stderr);
                                for line in reader.lines() {
                                    match line {
                                        Ok(l) if !l.is_empty() => log::warn!("[Node.js] {}", l),
                                        _ => {}
                                    }
                                }
                            });
                        }

                        let server_state = handle_node.state::<ServerProcess>();
                        *server_state.0.lock().unwrap() = Some(child);

                        let url = format!("http://127.0.0.1:{}", node_port);
                        if wait_for_server(node_port, Duration::from_secs(30)) {
                            log::info!("[Lifecycle] Node.js server ready on port {}", node_port);
                        } else {
                            log::error!("[Lifecycle] Node.js server did not respond in 30s — navigating anyway");
                        }
                        // ALWAYS navigate — never leave the window at about:blank.
                        // Even if the health check failed, the user sees a Next.js
                        // error page instead of an unexplainable black screen.
                        if let Some(window) = handle_node.get_webview_window("main") {
                            let _ = window.navigate(url.parse().unwrap());
                        }
                    }
                    Err(e) => {
                        log::error!("[Lifecycle] Failed to start Node.js server: {}", e);
                    }
                }
            });

            let handle_llama = app.handle().clone();
            thread::spawn(move || {
                if model::model_exists() {
                    let llama_state = handle_llama.state::<llama_server::LlamaProcess>();
                    match llama_server::start(&handle_llama, &llama_state, llama_port) {
                        Ok(port) => {
                            if llama_server::wait_ready(port, Duration::from_secs(120)) {
                                log::info!("[Lifecycle] llama-server ready on port {}", port);
                            } else {
                                log::error!("[Lifecycle] llama-server failed to become ready within 120 seconds");
                            }
                        }
                        Err(e) => log::warn!("[Lifecycle] Could not start llama-server: {}", e),
                    }
                } else {
                    log::info!("[Lifecycle] Model not found — will prompt user to download");
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Epito")
        .run(|app_handle, event| {
            match event {
                tauri::RunEvent::WindowEvent {
                    label,
                    event: tauri::WindowEvent::CloseRequested { .. },
                    ..
                } => {
                    if label == "main" {
                        if let Some(window) = app_handle.get_webview_window("main") {
                            save_current_window_state(&window);
                        }
                    }
                }
                tauri::RunEvent::Exit => {
                    perform_shutdown(app_handle);
                }
                tauri::RunEvent::ExitRequested { .. } => {}
                _ => {}
            }
        });
}
