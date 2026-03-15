#!/usr/bin/env node
/**
 * Prepares the Next.js standalone build for Tauri bundling.
 *
 * Optimizations (saves ~150-200MB vs naive copy):
 *   - Strips build artifacts from better-sqlite3 (saves ~22MB)
 *   - Removes cross-platform binaries from onnxruntime-node (saves ~60MB)
 *   - Removes duplicate WASM variants from tesseract.js-core (saves ~30MB)
 *   - Copies only current-platform @img/sharp bindings (saves ~15MB)
 *   - Strips .md, .txt, .map, tests, docs from native modules
 */
import { cpSync, existsSync, mkdirSync, rmSync, readdirSync, statSync } from 'fs';
import { join, resolve, basename } from 'path';

const ROOT = resolve(import.meta.dirname, '..');
const STANDALONE = join(ROOT, '.next', 'standalone');
const STANDALONE_MODULES = join(STANDALONE, 'node_modules');

if (!existsSync(STANDALONE)) {
  console.error('ERROR: .next/standalone not found. Run "next build" first.');
  process.exit(1);
}

// Remove stale src-tauri from standalone (Next.js copies the whole project)
const staleDir = join(STANDALONE, 'src-tauri');
if (existsSync(staleDir)) {
  rmSync(staleDir, { recursive: true, force: true });
  console.log('[prepare] Removed stale src-tauri/');
}

// ─── Platform detection ──────────────────────────────────────────────────────

const PLATFORM = process.platform;  // 'win32', 'darwin', 'linux'
const ARCH = process.arch;          // 'x64', 'arm64'

// onnxruntime-node platform dirs to KEEP (delete all others)
const ORT_KEEP = `${PLATFORM === 'win32' ? 'win32' : PLATFORM}/${ARCH}`;

// @img/sharp packages to keep (platform-specific)
const SHARP_PLATFORM = PLATFORM === 'win32' ? 'win32' : PLATFORM === 'darwin' ? 'darwin' : 'linux';
const SHARP_KEEP_PREFIXES = [
  `sharp-${SHARP_PLATFORM}-${ARCH}`,
  `sharp-libvips-${SHARP_PLATFORM}-${ARCH}`,
  'colour',  // always needed
];

console.log(`[prepare] Platform: ${PLATFORM}-${ARCH}`);

// ─── Copy native modules ─────────────────────────────────────────────────────

const NATIVE_MODULES = [
  'better-sqlite3',
  'sharp',
  'tesseract.js',
  'tesseract.js-core',
  '@xenova/transformers',
  'onnxruntime-node',
  'pdf-parse',
  'mammoth',
];

console.log('[prepare] Copying native modules...');
let totalSaved = 0;

for (const mod of NATIVE_MODULES) {
  const src = join(ROOT, 'node_modules', mod);
  const dest = join(STANDALONE_MODULES, mod);

  if (!existsSync(src)) {
    console.log(`  SKIP: ${mod} (not installed)`);
    continue;
  }

  try {
    mkdirSync(join(dest, '..'), { recursive: true });
    cpSync(src, dest, { recursive: true, force: true });
    console.log(`  OK: ${mod}`);
  } catch (err) {
    console.warn(`  WARN: ${mod}: ${err.message}`);
  }
}

// ─── Strip better-sqlite3 build artifacts (~22MB saved) ──────────────────────

const bsqlDest = join(STANDALONE_MODULES, 'better-sqlite3');
if (existsSync(bsqlDest)) {
  // Keep build/Release/better_sqlite3.node (the actual native binary)
  // Strip everything else: .o files, .a files, deps/, src/, test/
  for (const dir of ['deps', 'src', 'benchmark', 'test']) {
    const target = join(bsqlDest, dir);
    if (existsSync(target)) {
      const size = dirSize(target);
      rmSync(target, { recursive: true, force: true });
      totalSaved += size;
      console.log(`  STRIP: better-sqlite3/${dir}/ (${mb(size)})`);
    }
  }

  // Strip build artifacts (.o, .a, .mk) but keep .node binaries
  const buildRelease = join(bsqlDest, 'build', 'Release');
  if (existsSync(buildRelease)) {
    for (const f of readdirSync(buildRelease, { recursive: true, withFileTypes: false })) {
      // Keep only .node files, remove .o, .a, .target dirs
    }
    const objDir = join(buildRelease, 'obj.target');
    if (existsSync(objDir)) {
      const size = dirSize(objDir);
      rmSync(objDir, { recursive: true, force: true });
      totalSaved += size;
      console.log(`  STRIP: better-sqlite3/build/Release/obj.target/ (${mb(size)})`);
    }
    // Remove .a static libraries
    for (const f of readdirSync(buildRelease)) {
      if (f.endsWith('.a')) {
        const target = join(buildRelease, f);
        const size = statSync(target).size;
        rmSync(target);
        totalSaved += size;
        console.log(`  STRIP: better-sqlite3/build/Release/${f} (${mb(size)})`);
      }
    }
    // Remove test_extension.node (not needed at runtime)
    const testExt = join(buildRelease, 'test_extension.node');
    if (existsSync(testExt)) {
      const size = statSync(testExt).size;
      rmSync(testExt);
      totalSaved += size;
    }
  }

  // Ensure prebuilds are present (fallback for bindings module)
  const prebuildsSrc = join(ROOT, 'node_modules', 'better-sqlite3', 'prebuilds');
  if (existsSync(prebuildsSrc)) {
    const prebuildsDest = join(bsqlDest, 'prebuilds');
    cpSync(prebuildsSrc, prebuildsDest, { recursive: true, force: true });
    console.log('  OK: better-sqlite3 prebuilds');
  }
}

// ─── Strip onnxruntime-node cross-platform binaries (~60MB saved) ────────────

const ortBin = join(STANDALONE_MODULES, 'onnxruntime-node', 'bin', 'napi-v3');
if (existsSync(ortBin)) {
  for (const platform of readdirSync(ortBin)) {
    const platformDir = join(ortBin, platform);
    if (!statSync(platformDir).isDirectory()) continue;
    for (const arch of readdirSync(platformDir)) {
      const key = `${platform}/${arch}`;
      if (key !== ORT_KEEP) {
        const target = join(platformDir, arch);
        const size = dirSize(target);
        rmSync(target, { recursive: true, force: true });
        totalSaved += size;
        console.log(`  STRIP: onnxruntime-node/${key}/ (${mb(size)})`);
      }
    }
  }
}

// ─── Strip tesseract.js-core duplicate WASM (~30MB saved) ────────────────────
// Keep only the standard WASM files, remove the .wasm.js (Base64 embedded) duplicates
// and the non-simd variants (modern CPUs all support SIMD)

const tessCore = join(STANDALONE_MODULES, 'tesseract.js-core');
if (existsSync(tessCore)) {
  const wasmJsFiles = readdirSync(tessCore).filter(f => f.endsWith('.wasm.js'));
  for (const f of wasmJsFiles) {
    const target = join(tessCore, f);
    const size = statSync(target).size;
    rmSync(target);
    totalSaved += size;
    console.log(`  STRIP: tesseract.js-core/${f} (${mb(size)})`);
  }
  // Remove non-SIMD variants (all modern x64/arm64 CPUs have SIMD)
  const nonSimd = readdirSync(tessCore).filter(f =>
    f.endsWith('.wasm') && !f.includes('simd') && f.startsWith('tesseract-core')
  );
  for (const f of nonSimd) {
    const target = join(tessCore, f);
    const size = statSync(target).size;
    rmSync(target);
    totalSaved += size;
    console.log(`  STRIP: tesseract.js-core/${f} (${mb(size)})`);
  }
}

// ─── Copy only current-platform @img/sharp packages ──────────────────────────

const imgDir = join(ROOT, 'node_modules', '@img');
if (existsSync(imgDir)) {
  const imgDest = join(STANDALONE_MODULES, '@img');
  mkdirSync(imgDest, { recursive: true });
  for (const pkg of readdirSync(imgDir)) {
    const isForCurrentPlatform = SHARP_KEEP_PREFIXES.some(p => pkg.startsWith(p));
    if (!isForCurrentPlatform) {
      console.log(`  SKIP: @img/${pkg} (not for ${PLATFORM}-${ARCH})`);
      continue;
    }
    try {
      cpSync(join(imgDir, pkg), join(imgDest, pkg), { recursive: true, force: true });
      console.log(`  OK: @img/${pkg}`);
    } catch (err) {
      console.warn(`  WARN: @img/${pkg}: ${err.message}`);
    }
  }
}

// ─── Strip @xenova/transformers nested sharp (uses root sharp instead) ────────

const xenovaSharp = join(STANDALONE_MODULES, '@xenova', 'transformers', 'node_modules', 'sharp');
if (existsSync(xenovaSharp)) {
  const size = dirSize(xenovaSharp);
  rmSync(xenovaSharp, { recursive: true, force: true });
  totalSaved += size;
  console.log(`  STRIP: @xenova/transformers/node_modules/sharp/ (${mb(size)})`);
}

// ─── Strip .md, LICENSE, .map, tests, docs from ALL native modules ───────────

const STRIP_PATTERNS = ['.md', '.txt', '.map', '.ts', '.flow'];
const STRIP_DIRS = ['test', 'tests', '__tests__', 'docs', 'doc', 'example', 'examples', '.github'];

for (const mod of NATIVE_MODULES) {
  const modDir = join(STANDALONE_MODULES, mod.startsWith('@') ? mod : mod);
  if (!existsSync(modDir)) continue;
  stripJunk(modDir);
}

function stripJunk(dir) {
  if (!existsSync(dir)) return;
  try {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (STRIP_DIRS.includes(entry.name)) {
          const size = dirSize(full);
          rmSync(full, { recursive: true, force: true });
          totalSaved += size;
        } else {
          stripJunk(full);
        }
      } else if (entry.isFile()) {
        if (STRIP_PATTERNS.some(p => entry.name.endsWith(p)) && entry.name !== 'index.d.ts') {
          const size = statSync(full).size;
          rmSync(full);
          totalSaved += size;
        }
      }
    }
  } catch {}
}

// ─── Copy other assets ───────────────────────────────────────────────────────

const scriptsSrc = join(ROOT, 'scripts');
const scriptsDest = join(STANDALONE, 'scripts');
if (existsSync(scriptsSrc)) {
  cpSync(scriptsSrc, scriptsDest, { recursive: true, force: true });
  console.log('  OK: scripts/');
}

const staticSrc = join(ROOT, '.next', 'static');
const staticDest = join(STANDALONE, '.next', 'static');
if (existsSync(staticSrc)) {
  mkdirSync(join(staticDest, '..'), { recursive: true });
  cpSync(staticSrc, staticDest, { recursive: true, force: true });
  console.log('  OK: .next/static/');
}

const publicSrc = join(ROOT, 'public');
const publicDest = join(STANDALONE, 'public');
if (existsSync(publicSrc)) {
  cpSync(publicSrc, publicDest, { recursive: true, force: true });
  console.log('  OK: public/');
}

mkdirSync(join(STANDALONE, 'data', 'uploads'), { recursive: true });

const pdfjsWorkerSrc = join(ROOT, 'node_modules', 'pdfjs-dist', 'legacy', 'build', 'pdf.worker.mjs');
const pdfjsWorkerDest = join(STANDALONE_MODULES, 'pdfjs-dist', 'legacy', 'build', 'pdf.worker.mjs');
if (existsSync(pdfjsWorkerSrc) && !existsSync(pdfjsWorkerDest)) {
  mkdirSync(join(pdfjsWorkerDest, '..'), { recursive: true });
  cpSync(pdfjsWorkerSrc, pdfjsWorkerDest);
  console.log('  OK: pdfjs-dist worker');
}

// Copy llama-server shared libraries into standalone
const libDir = join(ROOT, 'src-tauri', 'binaries');
const libDest = join(STANDALONE, 'lib');
if (existsSync(libDir)) {
  const libExts = ['.dll', '.dylib', '.so'];
  const libs = readdirSync(libDir).filter(f => libExts.some(ext => f.endsWith(ext)));
  if (libs.length > 0) {
    mkdirSync(libDest, { recursive: true });
    for (const lib of libs) {
      try {
        cpSync(join(libDir, lib), join(libDest, lib), { force: true });
        console.log(`  OK: lib/${lib}`);
      } catch (err) {
        console.warn(`  WARN: ${lib}: ${err.message}`);
      }
    }
  }
}

// ─── Summary ─────────────────────────────────────────────────────────────────

console.log(`\n[prepare] ✓ Done. Stripped ${mb(totalSaved)} of unnecessary files.`);

function dirSize(dir) {
  let size = 0;
  try {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) size += dirSize(full);
      else if (entry.isFile()) size += statSync(full).size;
    }
  } catch {}
  return size;
}

function mb(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
