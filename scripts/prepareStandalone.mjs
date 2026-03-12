#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, rmSync, readdirSync } from 'fs';
import { join, resolve } from 'path';

const ROOT = resolve(import.meta.dirname, '..');
const STANDALONE = join(ROOT, '.next', 'standalone');
const STANDALONE_MODULES = join(STANDALONE, 'node_modules');

if (!existsSync(STANDALONE)) {
  console.error('ERROR: .next/standalone not found. Run "next build" with output: "standalone" first.');
  process.exit(1);
}

const staleDir = join(STANDALONE, 'src-tauri');
if (existsSync(staleDir)) {
  rmSync(staleDir, { recursive: true, force: true });
  console.log('[prepareStandalone] Removed stale src-tauri/ from standalone (prevents recursive bloat)');
}

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

console.log('[prepareStandalone] Copying native modules...');

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
    console.warn(`  WARN: Failed to copy ${mod}:`, err.message);
  }
}

const prebuildsSrc = join(ROOT, 'node_modules', 'better-sqlite3', 'prebuilds');
if (existsSync(prebuildsSrc)) {
  const prebuildsDest = join(STANDALONE_MODULES, 'better-sqlite3', 'prebuilds');
  cpSync(prebuildsSrc, prebuildsDest, { recursive: true, force: true });
  console.log('  OK: better-sqlite3 prebuilds');
}

// Copy platform-specific optional packages (e.g. @img/sharp-win32-x64).
// sharp v0.34+ splits native bindings into @img/sharp-{platform}-{arch}.
// These are separate npm packages that MUST be in node_modules at runtime.
const imgDir = join(ROOT, 'node_modules', '@img');
if (existsSync(imgDir)) {
  const imgDest = join(STANDALONE_MODULES, '@img');
  mkdirSync(imgDest, { recursive: true });
  for (const pkg of readdirSync(imgDir)) {
    const src = join(imgDir, pkg);
    const dest = join(imgDest, pkg);
    try {
      cpSync(src, dest, { recursive: true, force: true });
      console.log(`  OK: @img/${pkg}`);
    } catch (err) {
      console.warn(`  WARN: Failed to copy @img/${pkg}:`, err.message);
    }
  }
}

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
console.log('  OK: data/');

const pdfjsWorkerSrc = join(ROOT, 'node_modules', 'pdfjs-dist', 'legacy', 'build', 'pdf.worker.mjs');
const pdfjsWorkerDest = join(STANDALONE_MODULES, 'pdfjs-dist', 'legacy', 'build', 'pdf.worker.mjs');
if (existsSync(pdfjsWorkerSrc) && !existsSync(pdfjsWorkerDest)) {
  mkdirSync(join(pdfjsWorkerDest, '..'), { recursive: true });
  cpSync(pdfjsWorkerSrc, pdfjsWorkerDest);
  console.log('  OK: pdfjs-dist worker (pdf.worker.mjs)');
}

// Copy llama-server shared libraries (DLLs/dylibs/SOs) into standalone
// so they get bundled via the "../.next/standalone/**/*" resource glob.
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
        console.warn(`  WARN: Failed to copy ${lib}:`, err.message);
      }
    }
  } else {
    console.log('  SKIP: No shared libraries found in src-tauri/binaries/ (llama-server will need to be set up separately)');
  }
}

console.log('[prepareStandalone] Done.');
