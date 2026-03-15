#!/usr/bin/env node
import { existsSync, mkdirSync, chmodSync, createWriteStream, statSync, readdirSync, renameSync, copyFileSync, unlinkSync, rmSync } from 'fs';
import { join, resolve } from 'path';
import { execSync } from 'child_process';
import { pipeline } from 'stream/promises';

const ROOT = resolve(import.meta.dirname, '..');
const BIN_DIR = join(ROOT, 'src-tauri', 'binaries');

const LLAMA_CPP_VERSION = 'b4722';
const GITHUB_BASE = `https://github.com/ggerganov/llama.cpp/releases/download/${LLAMA_CPP_VERSION}`;

// ---------------------------------------------------------------------------
// GPU detection (runs at download time on developer/build machine)
// ---------------------------------------------------------------------------

function detectWindowsGpu() {
  // 1. Check for NVIDIA GPU via nvidia-smi
  try {
    const nvOut = execSync('nvidia-smi --query-gpu=name --format=csv,noheader', {
      encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], timeout: 5000,
    }).trim();
    if (nvOut) {
      console.log(`[GPU] Detected NVIDIA GPU: ${nvOut}`);
      return 'nvidia';
    }
  } catch {}

  // 2. Check via WMIC for any discrete GPU
  try {
    const wmicOut = execSync('wmic path win32_VideoController get name', {
      encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], timeout: 5000,
    }).trim();
    const lines = wmicOut.split('\n').map(l => l.trim()).filter(l => l && l !== 'Name');
    for (const line of lines) {
      if (/nvidia/i.test(line)) {
        console.log(`[GPU] Detected NVIDIA GPU (WMIC): ${line}`);
        return 'nvidia';
      }
      if (/radeon|amd/i.test(line)) {
        console.log(`[GPU] Detected AMD GPU: ${line}`);
        return 'amd';
      }
      if (/intel.*(?:arc|iris|uhd|hd)/i.test(line)) {
        console.log(`[GPU] Detected Intel GPU: ${line}`);
        return 'intel';
      }
    }
  } catch {}

  console.log('[GPU] No discrete GPU detected — using CPU build');
  return 'cpu';
}

function detectGpu() {
  if (process.platform === 'darwin') return 'metal'; // macOS always has Metal
  if (process.platform === 'win32') return detectWindowsGpu();

  // Linux
  try {
    execSync('nvidia-smi', { stdio: 'pipe', timeout: 5000 });
    return 'nvidia';
  } catch {}
  try {
    execSync('rocminfo', { stdio: 'pipe', timeout: 5000 });
    return 'amd';
  } catch {}
  return 'cpu';
}

// ---------------------------------------------------------------------------
// Platform + GPU → archive name mapping
// ---------------------------------------------------------------------------

function getArchiveName(platformKey, gpu) {
  const archives = {
    'darwin-arm64': `llama-${LLAMA_CPP_VERSION}-bin-macos-arm64.zip`,
    'darwin-x64':  `llama-${LLAMA_CPP_VERSION}-bin-macos-x64.zip`,
    'linux-x64':   `llama-${LLAMA_CPP_VERSION}-bin-ubuntu-x64.zip`,
  };

  if (archives[platformKey]) return archives[platformKey];

  if (platformKey === 'win32-x64') {
    // Pick the best build for the detected GPU:
    //   - NVIDIA → Vulkan build (universal, works great, no 574MB cudart needed)
    //   - AMD/Intel → Vulkan build
    //   - No GPU → CPU-only AVX2 build
    if (gpu === 'nvidia' || gpu === 'amd' || gpu === 'intel') {
      console.log(`[downloadLlamaServer] Using Vulkan build for ${gpu.toUpperCase()} GPU acceleration`);
      return `llama-${LLAMA_CPP_VERSION}-bin-win-vulkan-x64.zip`;
    }
    console.log('[downloadLlamaServer] Using CPU-only (AVX2) build');
    return `llama-${LLAMA_CPP_VERSION}-bin-win-avx2-x64.zip`;
  }

  return null;
}

const TRIPLES = {
  'darwin-arm64': 'aarch64-apple-darwin',
  'darwin-x64':   'x86_64-apple-darwin',
  'linux-x64':    'x86_64-unknown-linux-gnu',
  'win32-x64':    'x86_64-pc-windows-msvc',
};

const LIB_EXTS = {
  'darwin': '.dylib',
  'linux':  '.so',
  'win32':  '.dll',
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const platformKey = `${process.platform}-${process.arch}`;
const triple = TRIPLES[platformKey];
const dylibExt = LIB_EXTS[process.platform];

if (!triple) {
  console.error(`Unsupported platform: ${platformKey}`);
  process.exit(1);
}

const gpu = detectGpu();
const archiveName = getArchiveName(platformKey, gpu);
if (!archiveName) {
  console.error(`No archive available for ${platformKey}`);
  process.exit(1);
}

const ext = process.platform === 'win32' ? '.exe' : '';
const binaryName = `llama-server${ext}`;
const outputPath = join(BIN_DIR, `llama-server-${triple}${ext}`);

function moveFile(src, dest) {
  try { renameSync(src, dest); }
  catch { copyFileSync(src, dest); unlinkSync(src); }
}

function findFileRecursive(dir, filename) {
  if (!existsSync(dir)) return null;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isFile() && entry.name === filename) return full;
    if (entry.isDirectory()) {
      const found = findFileRecursive(full, filename);
      if (found) return found;
    }
  }
  return null;
}

function findAllByExt(dir, extension) {
  const results = [];
  if (!existsSync(dir)) return results;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isFile() && entry.name.endsWith(extension)) results.push(full);
    else if (entry.isDirectory()) results.push(...findAllByExt(full, extension));
  }
  return results;
}

// Skip if binary + libs already exist
if (existsSync(outputPath)) {
  const size = statSync(outputPath).size;
  if (size > 1000) {
    const hasLibs = readdirSync(BIN_DIR).some(f =>
      f.endsWith(dylibExt) && (process.platform === 'win32' || f.startsWith('lib'))
    );
    // Also check that we have the GPU DLL if we expect one
    const hasGpuDll = gpu === 'cpu' || gpu === 'metal' ||
      readdirSync(BIN_DIR).some(f => f === 'ggml-vulkan.dll' || f === 'ggml-cuda.dll' || f.endsWith('.dylib'));
    if (hasLibs && hasGpuDll) {
      console.log(`[downloadLlamaServer] Binary and libs already exist (${(size / 1024 / 1024).toFixed(1)} MB): ${outputPath}`);
      process.exit(0);
    }
    console.log('[downloadLlamaServer] Binary exists but libraries missing, re-downloading...');
  }
}

mkdirSync(BIN_DIR, { recursive: true });

const downloadUrl = `${GITHUB_BASE}/${archiveName}`;

console.log(`[downloadLlamaServer] Platform: ${platformKey} → ${triple}`);
console.log(`[downloadLlamaServer] GPU: ${gpu}`);
console.log(`[downloadLlamaServer] Archive: ${archiveName}`);
console.log(`[downloadLlamaServer] Downloading: ${downloadUrl}`);

try {
  const response = await fetch(downloadUrl, { redirect: 'follow' });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

  const zipPath = join(BIN_DIR, archiveName);
  const fileStream = createWriteStream(zipPath);
  await pipeline(response.body, fileStream);
  console.log(`[downloadLlamaServer] Downloaded: ${zipPath}`);

  const tmpDir = join(BIN_DIR, '_extract_tmp');
  rmSync(tmpDir, { recursive: true, force: true });
  mkdirSync(tmpDir, { recursive: true });

  if (process.platform === 'win32') {
    execSync(`tar -xf "${zipPath}" -C "${tmpDir}"`, { stdio: 'pipe' });
  } else {
    execSync(`unzip -o "${zipPath}" -d "${tmpDir}"`, { stdio: 'pipe' });
  }

  // Find and install the binary
  const foundBinary = findFileRecursive(tmpDir, binaryName);
  if (!foundBinary) throw new Error(`${binaryName} not found in archive`);
  console.log(`[downloadLlamaServer] Found binary: ${foundBinary}`);
  moveFile(foundBinary, outputPath);

  // Find and install ALL shared libraries
  const libs = findAllByExt(tmpDir, dylibExt);
  let libCount = 0;
  for (const libPath of libs) {
    const libName = libPath.split(/[\\/]/).pop();
    if (process.platform !== 'win32' && !libName.startsWith('lib')) continue;
    moveFile(libPath, join(BIN_DIR, libName));
    console.log(`[downloadLlamaServer] Installed lib: ${libName}`);
    libCount++;
  }

  if (process.platform !== 'win32') {
    chmodSync(outputPath, 0o755);
    for (const f of readdirSync(BIN_DIR)) {
      if (f.endsWith(dylibExt)) chmodSync(join(BIN_DIR, f), 0o755);
    }
  }

  rmSync(tmpDir, { recursive: true, force: true });
  rmSync(zipPath, { force: true });

  // Validate
  if (!existsSync(outputPath)) throw new Error(`Binary not created at: ${outputPath}`);
  const finalSize = statSync(outputPath).size;
  if (finalSize < 1000) throw new Error(`Binary too small (${finalSize} bytes)`);

  const installedLibs = readdirSync(BIN_DIR).filter(f => f.endsWith(dylibExt));
  console.log(`[downloadLlamaServer] Installed ${installedLibs.length} libs: ${installedLibs.join(', ')}`);

  // Report GPU status
  const hasVulkan = installedLibs.includes('ggml-vulkan.dll');
  const hasCuda = installedLibs.includes('ggml-cuda.dll');
  if (hasVulkan) console.log('[downloadLlamaServer] ✓ Vulkan GPU acceleration enabled');
  else if (hasCuda) console.log('[downloadLlamaServer] ✓ CUDA GPU acceleration enabled');
  else if (process.platform === 'win32') console.log('[downloadLlamaServer] ⚠ CPU-only build — no GPU acceleration');

  console.log(`[downloadLlamaServer] Done: ${outputPath} (${(finalSize / 1024 / 1024).toFixed(1)} MB)`);
} catch (err) {
  console.error(`[downloadLlamaServer] Error: ${err.message}`);
  console.error('');
  console.error('Manual download: https://github.com/ggml-org/llama.cpp/releases');
  console.error(`Place binary at: ${outputPath}`);
  process.exit(1);
}
