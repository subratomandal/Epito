use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use serde::Serialize;
use tauri::{AppHandle, Emitter};

const MODEL_FILENAME: &str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf";
const MODEL_URL: &str = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf";
const EXPECTED_SIZE: u64 = 4_368_438_944;

static DOWNLOAD_PROGRESS: Mutex<Option<DownloadProgress>> = Mutex::new(None);

#[derive(Clone, Serialize)]
pub struct ModelStatus {
    pub exists: bool,
    pub path: String,
    pub size_bytes: u64,
    pub downloading: bool,
}

#[derive(Clone, Serialize)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: u64,
    pub percent: f64,
    pub speed_mbps: f64,
}

pub fn models_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Cannot determine home directory")
        .join(".epito")
        .join("models")
}

pub fn model_path() -> PathBuf {
    models_dir().join(MODEL_FILENAME)
}

pub fn data_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Cannot determine home directory")
        .join(".epito")
        .join("data")
}

pub fn model_exists() -> bool {
    let path = model_path();
    if !path.exists() {
        return false;
    }
    match fs::metadata(&path) {
        Ok(meta) => {
            let size = meta.len();
            size > EXPECTED_SIZE / 2
        }
        Err(_) => false,
    }
}

fn partial_download_size() -> u64 {
    let tmp_path = model_path().with_extension("gguf.tmp");
    fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0)
}

#[tauri::command]
pub fn check_model() -> ModelStatus {
    let path = model_path();
    let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    ModelStatus {
        exists: model_exists(),
        path: path.to_string_lossy().to_string(),
        size_bytes: size,
        downloading: false,
    }
}

#[tauri::command]
pub async fn download_model(app: AppHandle) -> Result<String, String> {
    let dir = models_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create models dir: {}", e))?;

    let final_path = model_path();
    let tmp_path = final_path.with_extension("gguf.tmp");

    if model_exists() {
        return Ok(final_path.to_string_lossy().to_string());
    }

    for attempt in 0..3 {
        if attempt > 0 {
            log::info!("Retry attempt {} for model download", attempt + 1);
            tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempt as u32))).await;
        }

        let resume_from = partial_download_size();
        match do_download(&app, &tmp_path, &final_path, resume_from).await {
            Ok(path) => return Ok(path),
            Err(e) => {
                log::error!("Download attempt {} failed: {}", attempt + 1, e);
                if attempt == 2 {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(format!("Download failed after 3 attempts: {}", e));
                }
            }
        }
    }

    Err("Download failed".to_string())
}

async fn do_download(
    app: &AppHandle,
    tmp_path: &PathBuf,
    final_path: &PathBuf,
    resume_from: u64,
) -> Result<String, String> {
    let client = reqwest::Client::new();

    let mut request = client.get(MODEL_URL);

    if resume_from > 0 {
        log::info!("Resuming download from {} bytes", resume_from);
        request = request.header("Range", format!("bytes={}-", resume_from));
    }

    let response = request.send().await.map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() && response.status().as_u16() != 206 {
        return Err(format!("HTTP error: {}", response.status()));
    }

    let total_size = if response.status().as_u16() == 206 {
        response
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split('/').last())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(EXPECTED_SIZE)
    } else {
        response
            .content_length()
            .unwrap_or(EXPECTED_SIZE)
    };

    use tokio::io::AsyncWriteExt;
    let file = if resume_from > 0 {
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(tmp_path)
            .await
            .map_err(|e| format!("Failed to open tmp file for resume: {}", e))?
    } else {
        tokio::fs::File::create(tmp_path)
            .await
            .map_err(|e| format!("Failed to create tmp file: {}", e))?
    };

    let mut writer = tokio::io::BufWriter::new(file);
    let mut downloaded = resume_from;
    let start_time = std::time::Instant::now();
    let mut last_progress = std::time::Instant::now();

    use futures_util::StreamExt;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download stream error: {}", e))?;
        writer.write_all(&chunk).await.map_err(|e| format!("Write error: {}", e))?;
        downloaded += chunk.len() as u64;

        if last_progress.elapsed() > std::time::Duration::from_millis(500) {
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed = if elapsed > 0.0 {
                ((downloaded - resume_from) as f64) / elapsed / 1_048_576.0
            } else {
                0.0
            };

            let progress = DownloadProgress {
                downloaded,
                total: total_size,
                percent: if total_size > 0 {
                    (downloaded as f64 / total_size as f64 * 100.0).min(100.0)
                } else {
                    0.0
                },
                speed_mbps: speed,
            };

            if let Ok(mut guard) = DOWNLOAD_PROGRESS.lock() {
                *guard = Some(progress.clone());
            }
            let _ = app.emit("model-download-progress", &progress);
            last_progress = std::time::Instant::now();
        }
    }

    writer.flush().await.map_err(|e| format!("Flush error: {}", e))?;
    drop(writer);

    let final_size = tokio::fs::metadata(tmp_path)
        .await
        .map_err(|e| format!("Failed to read tmp file metadata: {}", e))?
        .len();

    if final_size < EXPECTED_SIZE / 2 {
        let _ = tokio::fs::remove_file(tmp_path).await;
        return Err(format!(
            "Downloaded file too small: {} bytes (expected ~{} bytes)",
            final_size, EXPECTED_SIZE
        ));
    }

    tokio::fs::rename(tmp_path, final_path)
        .await
        .map_err(|e| format!("Failed to rename tmp to final: {}", e))?;

    let _ = app.emit("model-download-progress", &DownloadProgress {
        downloaded: final_size,
        total: final_size,
        percent: 100.0,
        speed_mbps: 0.0,
    });

    if let Ok(mut guard) = DOWNLOAD_PROGRESS.lock() {
        *guard = None;
    }

    log::info!("Model download complete: {} bytes", final_size);
    Ok(final_path.to_string_lossy().to_string())
}

#[tauri::command]
pub fn get_download_progress() -> Option<DownloadProgress> {
    DOWNLOAD_PROGRESS.lock().ok().and_then(|g| g.clone())
}

#[tauri::command]
pub fn delete_model() -> Result<(), String> {
    let path = model_path();
    if path.exists() {
        fs::remove_file(&path).map_err(|e| format!("Failed to delete model: {}", e))?;
    }
    let tmp = path.with_extension("gguf.tmp");
    if tmp.exists() {
        let _ = fs::remove_file(&tmp);
    }
    Ok(())
}
