#!/usr/bin/env node
import { existsSync, mkdirSync, chmodSync, createWriteStream, statSync, readdirSync, renameSync, copyFileSync, unlinkSync, rmSync } from 'fs';
import { join, resolve } from 'path';
import { execSync } from 'child_process';
import { pipeline } from 'stream/promises';

const ROOT = resolve(import.meta.dirname, '..');
const BIN_DIR = join(ROOT, 'src-tauri', 'binaries');

const LLAMA_CPP_VERSION = 'b4722';

const PLATFORMS = {
  'darwin-arm64': {
    triple: 'aarch64-apple-darwin',
    archive: `llama-${LLAMA_CPP_VERSION}-bin-macos-arm64.zip`,
    dylibExt: '.dylib',
  },
  'darwin-x64': {
    triple: 'x86_64-apple-darwin',
    archive: `llama-${LLAMA_CPP_VERSION}-bin-macos-x64.zip`,
    dylibExt: '.dylib',
  },
  'linux-x64': {
    triple: 'x86_64-unknown-linux-gnu',
    archive: `llama-${LLAMA_CPP_VERSION}-bin-ubuntu-x64.zip`,
    dylibExt: '.so',
  },
  'win32-x64': {
    triple: 'x86_64-pc-windows-msvc',
    archive: `llama-${LLAMA_CPP_VERSION}-bin-win-avx2-x64.zip`,
    dylibExt: '.dll',
  },
};

const platformKey = `${process.platform}-${process.arch}`;
const config = PLATFORMS[platformKey];

if (!config) {
  console.error(`Unsupported platform: ${platformKey}`);
  console.error(`Supported: ${Object.keys(PLATFORMS).join(', ')}`);
  process.exit(1);
}

const ext = process.platform === 'win32' ? '.exe' : '';
const binaryName = `llama-server${ext}`;
const outputPath = join(BIN_DIR, `llama-server-${config.triple}${ext}`);

function moveFile(src, dest) {
  try {
    renameSync(src, dest);
  } catch {
    copyFileSync(src, dest);
    unlinkSync(src);
  }
}

// Recursively find a file by name anywhere in a directory tree
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

// Recursively find ALL files matching an extension
function findAllByExt(dir, extension) {
  const results = [];
  if (!existsSync(dir)) return results;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isFile() && entry.name.endsWith(extension)) {
      results.push(full);
    } else if (entry.isDirectory()) {
      results.push(...findAllByExt(full, extension));
    }
  }
  return results;
}

if (existsSync(outputPath)) {
  const size = statSync(outputPath).size;
  if (size > 1000) {
    // On Windows, DLLs don't have a "lib" prefix (e.g. ggml-base.dll, llama.dll).
    // On macOS/Linux, shared libs start with "lib" (e.g. libggml.dylib).
    const hasDylibs = readdirSync(BIN_DIR).some(f =>
      f.endsWith(config.dylibExt) && (process.platform === 'win32' || f.startsWith('lib'))
    );
    if (hasDylibs) {
      console.log(`[downloadLlamaServer] Binary and libs already exist (${(size / 1024 / 1024).toFixed(1)} MB): ${outputPath}`);
      process.exit(0);
    }
    console.log(`[downloadLlamaServer] Binary exists but dynamic libraries missing, re-downloading...`);
  } else {
    console.log(`[downloadLlamaServer] Existing binary is invalid (${size} bytes), re-downloading...`);
  }
}

mkdirSync(BIN_DIR, { recursive: true });

const downloadUrl = `https://github.com/ggerganov/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/${config.archive}`;
const zipPath = join(BIN_DIR, config.archive);

console.log(`[downloadLlamaServer] Platform: ${platformKey} → ${config.triple}`);
console.log(`[downloadLlamaServer] Downloading: ${downloadUrl}`);

try {
  const response = await fetch(downloadUrl, { redirect: 'follow' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const fileStream = createWriteStream(zipPath);
  await pipeline(response.body, fileStream);
  console.log(`[downloadLlamaServer] Downloaded: ${zipPath}`);

  const tmpDir = join(BIN_DIR, '_extract_tmp');
  rmSync(tmpDir, { recursive: true, force: true });
  mkdirSync(tmpDir, { recursive: true });

  // Extract entire archive
  if (process.platform === 'win32') {
    execSync(`tar -xf "${zipPath}" -C "${tmpDir}"`, { stdio: 'pipe' });
  } else {
    execSync(`unzip -o "${zipPath}" -d "${tmpDir}"`, { stdio: 'pipe' });
  }

  // --- Find the llama-server binary ---
  // The zip structure varies across releases:
  //   - Flat: llama-server.exe at root
  //   - Nested: build/bin/llama-server.exe or build/bin/Release/llama-server.exe
  // We search the entire extracted tree to handle all cases.
  const foundBinary = findFileRecursive(tmpDir, binaryName);
  if (!foundBinary) {
    throw new Error(`${binaryName} not found anywhere in the archive`);
  }
  console.log(`[downloadLlamaServer] Found binary: ${foundBinary}`);
  moveFile(foundBinary, outputPath);

  // --- Find ALL shared libraries ---
  // Same approach: search the entire extracted tree for .dll/.dylib/.so files.
  // This handles flat zips (DLLs at root) and nested zips (DLLs in build/bin/).
  const libs = findAllByExt(tmpDir, config.dylibExt);
  let libCount = 0;
  for (const libPath of libs) {
    const libName = libPath.split(/[\\/]/).pop();
    // On macOS/Linux, only copy libs starting with "lib" (skip test artifacts)
    if (process.platform !== 'win32' && !libName.startsWith('lib')) continue;
    moveFile(libPath, join(BIN_DIR, libName));
    console.log(`[downloadLlamaServer] Installed lib: ${libName}`);
    libCount++;
  }
  console.log(`[downloadLlamaServer] Installed ${libCount} shared libraries`);

  if (process.platform !== 'win32') {
    chmodSync(outputPath, 0o755);
    for (const f of readdirSync(BIN_DIR)) {
      if (f.endsWith(config.dylibExt)) {
        chmodSync(join(BIN_DIR, f), 0o755);
      }
    }
  }

  // Clean up
  rmSync(tmpDir, { recursive: true, force: true });
  rmSync(zipPath, { force: true });

  // Validate
  if (!existsSync(outputPath)) {
    throw new Error(`Binary was not created at expected path: ${outputPath}`);
  }
  const finalSize = statSync(outputPath).size;
  if (finalSize < 1000) {
    throw new Error(`Installed binary is too small (${finalSize} bytes), likely corrupted`);
  }

  // Verify DLLs were actually extracted
  const installedLibs = readdirSync(BIN_DIR).filter(f => f.endsWith(config.dylibExt));
  if (installedLibs.length === 0) {
    console.warn(`[downloadLlamaServer] WARNING: No shared libraries (${config.dylibExt}) were found in the archive!`);
    console.warn(`[downloadLlamaServer] llama-server may fail at runtime with "DLL not found" errors.`);
  } else {
    console.log(`[downloadLlamaServer] Verified ${installedLibs.length} libs in ${BIN_DIR}: ${installedLibs.join(', ')}`);
  }

  console.log(`[downloadLlamaServer] Installed: ${outputPath} (${(finalSize / 1024 / 1024).toFixed(1)} MB)`);
} catch (err) {
  console.error(`[downloadLlamaServer] Error: ${err.message}`);
  console.error('');
  console.error('You can manually download llama-server from:');
  console.error('  https://github.com/ggerganov/llama.cpp/releases');
  console.error(`  Place it at: ${outputPath}`);
  process.exit(1);
}
