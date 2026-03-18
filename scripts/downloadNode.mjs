#!/usr/bin/env node
// Copies the running Node.js binary into src-tauri/binaries for Tauri bundling.
// Uses process.execPath to guarantee ABI compatibility with compiled native modules.
import { existsSync, mkdirSync, chmodSync, statSync, copyFileSync } from 'fs';
import { join, resolve } from 'path';
import { execSync } from 'child_process';

const ROOT = resolve(import.meta.dirname, '..');
const BIN_DIR = join(ROOT, 'src-tauri', 'binaries');

const TRIPLES = {
  'darwin-arm64': 'aarch64-apple-darwin',
  'darwin-x64': 'x86_64-apple-darwin',
  'linux-x64': 'x86_64-unknown-linux-gnu',
  'win32-x64': 'x86_64-pc-windows-msvc',
};

const platformKey = `${process.platform}-${process.arch}`;
const triple = TRIPLES[platformKey];

if (!triple) {
  console.error(`[bundleNode] Unsupported platform: ${platformKey}`);
  console.error(`[bundleNode] Supported: ${Object.keys(TRIPLES).join(', ')}`);
  process.exit(1);
}

const ext = process.platform === 'win32' ? '.exe' : '';
const outputPath = join(BIN_DIR, `node-${triple}${ext}`);

if (existsSync(outputPath)) {
  const size = statSync(outputPath).size;
  if (size > 1_000_000) {
    try {
      const bundledVersion = execSync(`"${outputPath}" --version`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
      if (bundledVersion === process.version) {
        console.log(`[bundleNode] Binary already exists and matches ${process.version} (${(size / 1024 / 1024).toFixed(1)} MB)`);
        process.exit(0);
      }
      console.log(`[bundleNode] Version mismatch (bundled: ${bundledVersion}, system: ${process.version}), updating...`);
    } catch {
      console.log(`[bundleNode] Existing binary is invalid, replacing...`);
    }
  }
}

mkdirSync(BIN_DIR, { recursive: true });

const sourceNode = process.execPath;
console.log(`[bundleNode] Copying system Node.js: ${sourceNode}`);
console.log(`[bundleNode] Version: ${process.version}, arch: ${process.arch}`);

copyFileSync(sourceNode, outputPath);

if (process.platform !== 'win32') {
  chmodSync(outputPath, 0o755);
}

const finalSize = statSync(outputPath).size;
console.log(`[bundleNode] Installed: ${outputPath} (${(finalSize / 1024 / 1024).toFixed(1)} MB)`);
