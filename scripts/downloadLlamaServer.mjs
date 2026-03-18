#!/usr/bin/env node
// Downloads the correct llama-server build for the current platform and GPU.
// Fallback chain: CUDA 12.4 -> Vulkan -> CPU AVX2. macOS uses Metal, Linux uses Ubuntu build.
import { existsSync, mkdirSync, chmodSync, createWriteStream, statSync, readdirSync, renameSync, copyFileSync, unlinkSync, rmSync, writeFileSync, readFileSync } from 'fs';
import { join, resolve } from 'path';
import { execSync } from 'child_process';
import { pipeline } from 'stream/promises';

const ROOT = resolve(import.meta.dirname, '..');
const BIN_DIR = join(ROOT, 'src-tauri', 'binaries');
const LLAMA_CPP_VERSION = 'b8340';
const GITHUB_BASE = `https://github.com/ggerganov/llama.cpp/releases/download/${LLAMA_CPP_VERSION}`;

const STAMP_FILE = join(BIN_DIR, '.llama-server-stamp.json');

// ─── GPU Detection ───────────────────────────────────────────────────────────

function detectGpu() {
  if (process.platform === 'darwin') return 'metal';
  if (process.platform !== 'win32') {
    try { execSync('nvidia-smi', { stdio: 'pipe', timeout: 5000 }); return 'nvidia'; } catch {}
    try { execSync('rocminfo', { stdio: 'pipe', timeout: 5000 }); return 'amd'; } catch {}
    return 'cpu';
  }

  // Windows GPU detection: nvidia-smi -> PowerShell -> WMIC fallback
  try {
    const out = execSync('nvidia-smi --query-gpu=name --format=csv,noheader', {
      encoding: 'utf8', stdio: 'pipe', timeout: 5000,
    }).trim();
    if (out) { console.log(`[GPU] NVIDIA (nvidia-smi): ${out}`); return 'nvidia'; }
  } catch {}

  try {
    const out = execSync(
      'powershell -NoProfile -NoLogo -Command "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name"',
      { encoding: 'utf8', stdio: 'pipe', timeout: 10000 }
    );
    for (const line of out.split('\n').map(l => l.trim()).filter(Boolean)) {
      console.log(`[GPU] Found (PowerShell): ${line}`);
      if (/nvidia|geforce|rtx|gtx|quadro/i.test(line)) return 'nvidia';
      if (/radeon|amd/i.test(line)) return 'amd';
      if (/intel.*(?:arc|iris|uhd|hd)/i.test(line)) return 'intel';
    }
  } catch {}

  try {
    const out = execSync('wmic path win32_VideoController get name', {
      encoding: 'utf8', stdio: 'pipe', timeout: 5000,
    });
    for (const line of out.split('\n').map(l => l.trim()).filter(l => l && l !== 'Name')) {
      console.log(`[GPU] Found (WMIC): ${line}`);
      if (/nvidia|geforce|rtx|gtx|quadro/i.test(line)) return 'nvidia';
      if (/radeon|amd/i.test(line)) return 'amd';
      if (/intel.*(?:arc|iris|uhd|hd)/i.test(line)) return 'intel';
    }
  } catch {}

  console.log('[GPU] No GPU detected by any method');
  return 'cpu';
}

// Returns the max CUDA version supported by the NVIDIA driver, or {0, 0} if unknown.
function detectCudaVersion() {
  try {
    const out = execSync('nvidia-smi', { encoding: 'utf8', stdio: 'pipe', timeout: 5000 });
    const match = out.match(/CUDA Version:\s*(\d+)\.(\d+)/);
    if (match) {
      const major = parseInt(match[1], 10);
      const minor = parseInt(match[2], 10);
      console.log(`[GPU] Driver CUDA version: ${major}.${minor}`);
      return { major, minor };
    }
  } catch {}
  console.log('[GPU] Could not detect CUDA version from nvidia-smi');
  return { major: 0, minor: 0 };
}

// ─── Archive Selection ───────────────────────────────────────────────────────

function getDownloads(platformKey, gpu) {
  const v = LLAMA_CPP_VERSION;

  const ARCHIVES = {
    'darwin-arm64': [{ url: `${GITHUB_BASE}/llama-${v}-bin-macos-arm64.zip`, desc: 'macOS ARM64 (Metal)' }],
    'darwin-x64':  [{ url: `${GITHUB_BASE}/llama-${v}-bin-macos-x64.zip`, desc: 'macOS x64' }],
    'linux-x64':   [{ url: `${GITHUB_BASE}/llama-${v}-bin-ubuntu-x64.zip`, desc: 'Linux x64' }],
  };

  if (ARCHIVES[platformKey]) return { downloads: ARCHIVES[platformKey], cudaTag: '' };

  if (platformKey === 'win32-x64') {
    if (gpu === 'nvidia') {
      const cuda = detectCudaVersion();

      // Always use CUDA 12.4: 13.x has known bugs on Ada Lovelace (RTX 40 series),
      // and 12.4 is forward-compatible with any driver advertising CUDA 12.0+.
      if (cuda.major >= 12) {
        const cudaTag = '12.4';
        console.log(`[GPU] Driver supports CUDA ${cuda.major}.${cuda.minor} → using CUDA 12.4 build (maximum compatibility)`);
        return {
          downloads: [
            { url: `${GITHUB_BASE}/llama-${v}-bin-win-cuda-${cudaTag}-x64.zip`, desc: `Windows CUDA ${cudaTag} (NVIDIA)` },
            { url: `${GITHUB_BASE}/cudart-llama-bin-win-cuda-${cudaTag}-x64.zip`, desc: `CUDA ${cudaTag} Runtime DLLs` },
          ],
          cudaTag,
        };
      }

      console.log(`[GPU] Driver CUDA ${cuda.major}.${cuda.minor} too old for CUDA 12.4 → falling back to Vulkan`);
    }
    if (gpu === 'amd' || gpu === 'intel') {
      return {
        downloads: [{ url: `${GITHUB_BASE}/llama-${v}-bin-win-vulkan-x64.zip`, desc: 'Windows Vulkan (AMD/Intel)' }],
        cudaTag: '',
      };
    }
    return {
      downloads: [{ url: `${GITHUB_BASE}/llama-${v}-bin-win-avx2-x64.zip`, desc: 'Windows CPU (AVX2)' }],
      cudaTag: '',
    };
  }

  return null;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const TRIPLES = {
  'darwin-arm64': 'aarch64-apple-darwin',
  'darwin-x64':   'x86_64-apple-darwin',
  'linux-x64':    'x86_64-unknown-linux-gnu',
  'win32-x64':    'x86_64-pc-windows-msvc',
};

const LIB_EXT = { darwin: '.dylib', linux: '.so', win32: '.dll' }[process.platform] || '.so';

function moveFile(src, dest) {
  try { renameSync(src, dest); } catch { copyFileSync(src, dest); unlinkSync(src); }
}

function findFileRecursive(dir, filename) {
  if (!existsSync(dir)) return null;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, e.name);
    if (e.isFile() && e.name === filename) return full;
    if (e.isDirectory()) { const f = findFileRecursive(full, filename); if (f) return f; }
  }
  return null;
}

function findAllByExt(dir, ext) {
  const r = [];
  if (!existsSync(dir)) return r;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, e.name);
    if (e.isFile() && e.name.endsWith(ext)) r.push(full);
    else if (e.isDirectory()) r.push(...findAllByExt(full, ext));
  }
  return r;
}

async function downloadAndExtract(url, desc, tmpDir) {
  console.log(`[download] ${desc}`);
  console.log(`[download] URL: ${url}`);

  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${desc}`);

  const zipPath = join(tmpDir, 'archive.zip');
  await pipeline(response.body, createWriteStream(zipPath));

  const extractDir = join(tmpDir, 'extract_' + Math.random().toString(36).slice(2));
  mkdirSync(extractDir, { recursive: true });

  if (process.platform === 'win32') {
    // PowerShell Expand-Archive: Windows tar misinterprets drive letters (C:) as remote hosts
    execSync(`powershell -NoProfile -Command "Expand-Archive -Path '${zipPath}' -DestinationPath '${extractDir}' -Force"`, { stdio: 'pipe', timeout: 120000 });
  } else {
    execSync(`unzip -o "${zipPath}" -d "${extractDir}"`, { stdio: 'pipe' });
  }

  rmSync(zipPath, { force: true });
  return extractDir;
}

// ─── Main ────────────────────────────────────────────────────────────────────

const platformKey = `${process.platform}-${process.arch}`;
const triple = TRIPLES[platformKey];
if (!triple) { console.error(`Unsupported: ${platformKey}`); process.exit(1); }

const gpu = detectGpu();
const result = getDownloads(platformKey, gpu);
if (!result) { console.error(`No archive for ${platformKey}`); process.exit(1); }
const { downloads, cudaTag } = result;

const ext = process.platform === 'win32' ? '.exe' : '';
const binaryName = `llama-server${ext}`;
const outputPath = join(BIN_DIR, `llama-server-${triple}${ext}`);

if (existsSync(STAMP_FILE) && existsSync(outputPath) && statSync(outputPath).size > 1000) {
  try {
    const stamp = JSON.parse(readFileSync(STAMP_FILE, 'utf8'));
    if (stamp.version === LLAMA_CPP_VERSION && stamp.gpu === gpu && stamp.cudaTag === cudaTag) {
      console.log(`[downloadLlamaServer] Already installed: ${LLAMA_CPP_VERSION} (${gpu}${cudaTag ? ` CUDA ${cudaTag}` : ''})`);
      process.exit(0);
    }
    console.log(`[downloadLlamaServer] Upgrading: ${stamp.version} → ${LLAMA_CPP_VERSION} (${gpu}${cudaTag ? ` CUDA ${cudaTag}` : ''})`);
  } catch {}
}

mkdirSync(BIN_DIR, { recursive: true });

console.log(`[downloadLlamaServer] Platform: ${platformKey} | GPU: ${gpu} | Downloads: ${downloads.length}`);

// Remove old shared libraries to prevent ABI mismatch from mixed versions
const oldLibs = readdirSync(BIN_DIR).filter(f => f.endsWith(LIB_EXT));
for (const lib of oldLibs) {
  rmSync(join(BIN_DIR, lib), { force: true });
  console.log(`[downloadLlamaServer] Removed old: ${lib}`);
}

try {
  const tmpDir = join(BIN_DIR, '_download_tmp');
  rmSync(tmpDir, { recursive: true, force: true });
  mkdirSync(tmpDir, { recursive: true });

  const extractDirs = [];
  for (const dl of downloads) {
    const dir = await downloadAndExtract(dl.url, dl.desc, tmpDir);
    extractDirs.push(dir);
  }

  const foundBinary = findFileRecursive(extractDirs[0], binaryName);
  if (!foundBinary) throw new Error(`${binaryName} not found in archive`);
  moveFile(foundBinary, outputPath);
  console.log(`[downloadLlamaServer] Installed binary: ${outputPath}`);

  let libCount = 0;
  for (const dir of extractDirs) {
    for (const libPath of findAllByExt(dir, LIB_EXT)) {
      const libName = libPath.split(/[\\/]/).pop();
      if (process.platform !== 'win32' && !libName.startsWith('lib')) continue;
      moveFile(libPath, join(BIN_DIR, libName));
      console.log(`[downloadLlamaServer] Installed lib: ${libName}`);
      libCount++;
    }
  }

  if (process.platform !== 'win32') {
    chmodSync(outputPath, 0o755);
    for (const f of readdirSync(BIN_DIR)) {
      if (f.endsWith(LIB_EXT)) chmodSync(join(BIN_DIR, f), 0o755);
    }
  }

  rmSync(tmpDir, { recursive: true, force: true });

  const finalSize = statSync(outputPath).size;
  if (finalSize < 1000) throw new Error(`Binary too small: ${finalSize} bytes`);

  const installedLibs = readdirSync(BIN_DIR).filter(f => f.endsWith(LIB_EXT));

  writeFileSync(STAMP_FILE, JSON.stringify({
    version: LLAMA_CPP_VERSION,
    gpu,
    cudaTag,
    installedAt: new Date().toISOString(),
    libs: installedLibs,
  }, null, 2));

  console.log(`\n[downloadLlamaServer] ═══════════════════════════════════════`);
  console.log(`[downloadLlamaServer] Version:   ${LLAMA_CPP_VERSION}`);
  console.log(`[downloadLlamaServer] Platform:  ${platformKey}`);
  console.log(`[downloadLlamaServer] GPU:       ${gpu.toUpperCase()}`);
  console.log(`[downloadLlamaServer] Binary:    ${outputPath} (${(finalSize / 1024 / 1024).toFixed(1)} MB)`);
  console.log(`[downloadLlamaServer] Libraries: ${libCount} (${installedLibs.join(', ')})`);

  if (installedLibs.some(f => f.includes('cuda')))
    console.log(`[downloadLlamaServer] Backend:   CUDA ${cudaTag} (NVIDIA GPU — maximum performance)`);
  else if (installedLibs.some(f => f.includes('vulkan')))
    console.log(`[downloadLlamaServer] Backend:   Vulkan (GPU accelerated)`);
  else if (process.platform === 'darwin')
    console.log(`[downloadLlamaServer] Backend:   Metal (Apple GPU)`);
  else
    console.log(`[downloadLlamaServer] Backend:   CPU only`);

  console.log(`[downloadLlamaServer] ═══════════════════════════════════════\n`);
} catch (err) {
  console.error(`[downloadLlamaServer] Error: ${err.message}`);
  console.error(`\nManual download: https://github.com/ggml-org/llama.cpp/releases/tag/${LLAMA_CPP_VERSION}`);
  console.error(`Place binary at: ${outputPath}`);
  process.exit(1);
}
