'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

interface DownloadProgress {
  downloaded: number;
  total: number;
  percent: number;
  speed_mbps: number;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function invoke<T = unknown>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const tauri = (window as any).__TAURI_INTERNALS__;
  if (!tauri?.invoke) throw new Error('Not running inside Tauri');
  return tauri.invoke(cmd, args);
}

function isTauriEnv(): boolean {
  return typeof window !== 'undefined' && !!(window as any).__TAURI_INTERNALS__;
}

export default function StartupScreen({ onReady }: { onReady: () => void }) {
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress>({ downloaded: 0, total: 0, percent: 0, speed_mbps: 0 });
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startup = useCallback(async () => {
    if (!isTauriEnv()) {
      onReady();
      return;
    }

    try {
      const status = await invoke<{ exists: boolean; path: string; size_bytes: number; downloading: boolean }>('check_model');

      if (!status.exists) {
        setDownloading(true);

        pollRef.current = setInterval(async () => {
          try {
            const prog = await invoke<DownloadProgress | null>('get_download_progress');
            if (prog) {
              setProgress(prev => ({
                ...prog,
                downloaded: Math.max(prev.downloaded, prog.downloaded),
                percent: Math.max(prev.percent, prog.percent),
              }));
            }
          } catch {}
        }, 500);

        await invoke('download_model');
        stopPolling();
        setDownloading(false);

        // Model just downloaded — start llama-server now.
        // The Rust startup thread already exited (model didn't exist at launch),
        // so we must explicitly trigger a lazy start.
        try {
          await invoke('start_llama_lazy');
        } catch (e) {
          console.warn('[Startup] llama-server start deferred:', e);
          // Non-fatal: server may take time to load the model.
          // AI features will become available once it's ready.
        }
      }

      onReady();
    } catch (err: any) {
      stopPolling();
      console.error('Startup error:', err);
      setError(err.message || err || 'Unknown error during startup');
    }
  }, [onReady, stopPolling]);

  useEffect(() => {
    startup();
    return () => stopPolling();
  }, [startup, stopPolling]);

  if (!downloading && !error) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-6 max-w-md px-8 text-center">
        {/* App title removed from startup */}

        {downloading && (
          <div className="flex flex-col items-center gap-4 w-full">
            <p className="text-sm text-muted-foreground">
              Downloading AI model ({formatBytes(progress.total || 4_370_000_000)})
            </p>
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-foreground rounded-full transition-[width] duration-300 ease-out"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
            <div className="flex justify-between w-full text-xs text-muted-foreground">
              <span>{formatBytes(progress.downloaded)}</span>
              <span>{progress.percent.toFixed(1)}%</span>
            </div>
            {progress.speed_mbps > 0 && (
              <p className="text-xs text-muted-foreground/60">
                {progress.speed_mbps.toFixed(1)} MB/s
              </p>
            )}
            <p className="text-xs text-muted-foreground/60">
              This is a one-time download. The model will be stored locally.
            </p>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center">
              <span className="text-red-500 text-lg">!</span>
            </div>
            <p className="text-sm text-red-500">{error}</p>
            <button
              onClick={() => {
                setError(null);
                startup();
              }}
              className="px-4 py-2 text-sm rounded-md bg-foreground text-background hover:opacity-90 transition-opacity"
            >
              Retry
            </button>
            <button
              onClick={onReady}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              Skip and continue anyway
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
