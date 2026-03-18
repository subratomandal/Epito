import { closeDb } from '@/notes/database';
import { closeBrowser } from '@/inference/browser';

let shutdownStarted = false;

const activeTaskCount = { value: 0 };
const MAX_CONCURRENT_TASKS = 5;

export function canAcceptTask(): boolean {
  return activeTaskCount.value < MAX_CONCURRENT_TASKS && !shutdownStarted;
}

export function taskStarted(): void {
  activeTaskCount.value++;
}

export function taskCompleted(): void {
  activeTaskCount.value = Math.max(0, activeTaskCount.value - 1);
}

export function isShuttingDown(): boolean {
  return shutdownStarted;
}

function performShutdown(signal: string): void {
  if (shutdownStarted) return;
  shutdownStarted = true;

  console.log(`[Lifecycle] Node.js shutdown initiated (${signal})`);

  try {
    closeBrowser().catch(() => {});
    closeDb();
    console.log('[Lifecycle] Database and browser closed.');
  } catch (err) {
    console.error('[Lifecycle] Error during shutdown:', err);
  }

  console.log('[Lifecycle] Node.js cleanup complete.');
}

export function installShutdownHandlers(): void {
  // SIGINT works on Windows (mapped to Ctrl+C) and Unix
  process.on('SIGINT', () => {
    performShutdown('SIGINT');
    process.exit(0);
  });

  // SIGTERM works on Unix; on Windows it only fires from process.kill()
  process.on('SIGTERM', () => {
    performShutdown('SIGTERM');
    process.exit(0);
  });

  // SIGQUIT is Unix-only; harmless on Windows (listener installs but never fires)
  if (process.platform !== 'win32') {
    process.on('SIGQUIT' as NodeJS.Signals, () => {
      performShutdown('SIGQUIT');
      process.exit(0);
    });
  }

  process.on('beforeExit', () => {
    performShutdown('beforeExit');
  });

  // 'exit' fires synchronously during any process.exit() call.
  // On Windows, this is the last chance to close the DB when Ctrl+C
  // triggers SIGINT → process.exit(0) → exit event.
  process.on('exit', () => {
    performShutdown('exit');
  });

  process.on('uncaughtException', (err) => {
    console.error('[Lifecycle] Uncaught exception:', err);
    performShutdown('uncaughtException');
    process.exit(1);
  });

  console.log('[Lifecycle] Node.js shutdown handlers installed');
}
