// Persistent headless browser for PDF/image export.
// Uses system Chrome/Edge — no bundled browser binary.
// Singleton: launches once, reuses for all exports, auto-closes after idle.

import { execSync } from 'child_process';
import { existsSync } from 'fs';

let browser: import('puppeteer-core').Browser | null = null;
let launchPromise: Promise<import('puppeteer-core').Browser> | null = null;
let idleTimer: ReturnType<typeof setTimeout> | null = null;
const IDLE_MS = 30_000;

// Find Chrome/Edge on the system. On Windows, Edge is always present (Tauri requires WebView2).
function findBrowser(): string | null {
  if (process.platform === 'darwin') {
    const paths = [
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
      '/Applications/Chromium.app/Contents/MacOS/Chromium',
      '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
      '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
    ];
    for (const p of paths) {
      if (existsSync(p)) return p;
    }
    // Try `which` as last resort
    try {
      return execSync('which google-chrome || which chromium', { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }).trim() || null;
    } catch { return null; }
  }

  if (process.platform === 'win32') {
    const paths = [
      process.env['PROGRAMFILES(X86)'] + '\\Microsoft\\Edge\\Application\\msedge.exe',
      process.env['PROGRAMFILES'] + '\\Microsoft\\Edge\\Application\\msedge.exe',
      process.env['PROGRAMFILES'] + '\\Google\\Chrome\\Application\\chrome.exe',
      process.env['PROGRAMFILES(X86)'] + '\\Google\\Chrome\\Application\\chrome.exe',
      process.env['LOCALAPPDATA'] + '\\Google\\Chrome\\Application\\chrome.exe',
      process.env['LOCALAPPDATA'] + '\\Microsoft\\Edge\\Application\\msedge.exe',
    ];
    for (const p of paths) {
      if (p && existsSync(p)) return p;
    }
    return null;
  }

  // Linux
  const names = ['google-chrome', 'google-chrome-stable', 'chromium-browser', 'chromium', 'microsoft-edge'];
  for (const name of names) {
    try {
      const p = execSync(`which ${name}`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
      if (p) return p;
    } catch { /* not found */ }
  }
  return null;
}

let cachedPath: string | null | undefined;

export function getBrowserPath(): string | null {
  if (cachedPath !== undefined) return cachedPath;
  cachedPath = findBrowser();
  if (cachedPath) console.log(`[Export] Browser found: ${cachedPath}`);
  else console.log('[Export] No Chrome/Edge found — PDF export will use fallback');
  return cachedPath;
}

export async function getBrowser(): Promise<import('puppeteer-core').Browser> {
  if (browser?.connected) {
    resetIdle();
    return browser;
  }

  if (launchPromise) return launchPromise;

  launchPromise = (async () => {
    const executablePath = getBrowserPath();
    if (!executablePath) throw new Error('No Chrome or Edge browser found on this system');

    const puppeteer = await import('puppeteer-core');
    browser = await puppeteer.default.launch({
      executablePath,
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        '--disable-extensions',
        '--disable-background-networking',
        '--disable-sync',
        '--disable-translate',
        '--hide-scrollbars',
        '--mute-audio',
        '--no-first-run',
      ],
    });

    console.log('[Export] Headless browser launched');
    resetIdle();
    return browser;
  })();

  try {
    return await launchPromise;
  } catch (err) {
    launchPromise = null;
    throw err;
  }
}

function resetIdle() {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(closeBrowser, IDLE_MS);
}

export async function closeBrowser() {
  if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
  if (browser) {
    try { await browser.close(); } catch {}
    browser = null;
    launchPromise = null;
    console.log('[Export] Headless browser closed (idle)');
  }
}

export function isBrowserAvailable(): boolean {
  return getBrowserPath() !== null;
}
