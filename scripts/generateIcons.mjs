#!/usr/bin/env node
// Generates circular logo icons for all platforms from a 1024x1024 master.
// Lanczos3 downscale + adaptive sharpening + anti-aliased circular mask.
import sharp from 'sharp';
import { mkdirSync, writeFileSync, existsSync } from 'fs';
import { join, resolve } from 'path';
import { execSync } from 'child_process';

const ROOT = resolve(import.meta.dirname, '..');
const PUBLIC_DIR = join(ROOT, 'public');
const ICONS_DIR = join(ROOT, 'src-tauri', 'icons');

const SOURCE_IMG = join(ROOT, 'assets', 'iconSource.png');
const MASTER_SIZE = 1024;

// ─── Master Source ────────────────────────────────────────────────────────

async function buildMasterSource() {
  if (existsSync(SOURCE_IMG)) {
    const meta = await sharp(SOURCE_IMG).metadata();
    console.log(`[generateIcons] Source: ${SOURCE_IMG} (${meta.width}x${meta.height})`);

    const master = await sharp(SOURCE_IMG)
      .resize(MASTER_SIZE, MASTER_SIZE, {
        fit: 'cover',
        kernel: sharp.kernel.lanczos3,
      })
      .sharpen({ sigma: 0.8, m1: 1.0, m2: 0.5 })
      .removeAlpha()
      .ensureAlpha()
      .png({ quality: 100, compressionLevel: 0 })
      .toBuffer();

    console.log(`[generateIcons] Master: ${MASTER_SIZE}x${MASTER_SIZE} (upscaled + sharpened)`);
    return master;
  }

  const SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
    <circle cx="512" cy="512" r="512" fill="#0a0a0f"/>
    <text x="512" y="740" font-family="SF Pro Display, Inter, Helvetica Neue, Arial, sans-serif" font-size="680" font-weight="800" fill="#ffffff" text-anchor="middle" letter-spacing="-16">E</text>
  </svg>`;
  const master = await sharp(Buffer.from(SVG))
    .resize(MASTER_SIZE, MASTER_SIZE)
    .png({ quality: 100, compressionLevel: 0 })
    .toBuffer();

  console.log('[generateIcons] Master: 1024x1024 (from SVG, pixel-perfect)');
  return master;
}

// ─── Circular PNG Generation ──────────────────────────────────────────────

async function generateRoundPNG(masterBuffer, size, outputPath) {
  // Adaptive sharpening: stronger for small sizes to keep the "E" crisp
  const sigma = size <= 32 ? 1.2 : size <= 64 ? 1.0 : size <= 128 ? 0.7 : 0.5;

  const resized = await sharp(masterBuffer)
    .resize(size, size, {
      fit: 'cover',
      kernel: sharp.kernel.lanczos3,
    })
    .sharpen({ sigma })
    .png({ compressionLevel: 6, adaptiveFiltering: true })
    .toBuffer();

  // 2x mask downscaled for anti-aliased edges
  const maskSize = Math.min(size * 2, 2048);
  const maskR = Math.floor(maskSize / 2);
  const circleMask = await sharp(
    Buffer.from(`<svg width="${maskSize}" height="${maskSize}"><circle cx="${maskR}" cy="${maskR}" r="${maskR}" fill="white"/></svg>`)
  )
    .resize(size, size, { kernel: sharp.kernel.lanczos3 })
    .toBuffer();

  await sharp(resized)
    .composite([{ input: circleMask, blend: 'dest-in' }])
    .png({ compressionLevel: 6, adaptiveFiltering: true })
    .toFile(outputPath);

  console.log(`  OK: ${outputPath.split('/').pop()} (${size}x${size})`);
}

async function generateRoundBuffer(masterBuffer, size) {
  const sigma = size <= 32 ? 1.2 : size <= 64 ? 1.0 : size <= 128 ? 0.7 : 0.5;

  const resized = await sharp(masterBuffer)
    .resize(size, size, {
      fit: 'cover',
      kernel: sharp.kernel.lanczos3,
    })
    .sharpen({ sigma })
    .png({ compressionLevel: 0 })
    .toBuffer();

  const maskSize = Math.min(size * 2, 2048);
  const maskR = Math.floor(maskSize / 2);
  const circleMask = await sharp(
    Buffer.from(`<svg width="${maskSize}" height="${maskSize}"><circle cx="${maskR}" cy="${maskR}" r="${maskR}" fill="white"/></svg>`)
  )
    .resize(size, size, { kernel: sharp.kernel.lanczos3 })
    .toBuffer();

  return sharp(resized)
    .composite([{ input: circleMask, blend: 'dest-in' }])
    .png({ compressionLevel: 0 })
    .toBuffer();
}

// ─── Main ────────────────────────────────────────────────────────────────

async function main() {
  console.log('[generateIcons] Creating high-quality circular icons...\n');

  const master = await buildMasterSource();

  const tauriSizes = [
    { size: 32, name: '32x32.png' },
    { size: 128, name: '128x128.png' },
    { size: 256, name: '128x128@2x.png' },
    { size: 256, name: 'icon.png' },
    { size: 30, name: 'Square30x30Logo.png' },
    { size: 44, name: 'Square44x44Logo.png' },
    { size: 71, name: 'Square71x71Logo.png' },
    { size: 89, name: 'Square89x89Logo.png' },
    { size: 107, name: 'Square107x107Logo.png' },
    { size: 142, name: 'Square142x142Logo.png' },
    { size: 150, name: 'Square150x150Logo.png' },
    { size: 284, name: 'Square284x284Logo.png' },
    { size: 310, name: 'Square310x310Logo.png' },
    { size: 50, name: 'StoreLogo.png' },
  ];

  for (const { size, name } of tauriSizes) {
    await generateRoundPNG(master, size, join(ICONS_DIR, name));
  }

  await generateRoundPNG(master, 512, join(PUBLIC_DIR, 'logo.png'));
  await generateRoundPNG(master, 192, join(PUBLIC_DIR, 'icon-192.png'));
  await generateRoundPNG(master, 512, join(PUBLIC_DIR, 'icon-512.png'));
  await generateRoundPNG(master, 32, join(PUBLIC_DIR, 'favicon.png'));

  // All sizes Windows needs at various DPI levels to avoid blurry scaling
  const icoSizes = [16, 24, 32, 48, 64, 128, 256];
  const icoPngs = [];
  console.log('');
  for (const size of icoSizes) {
    const buf = await generateRoundBuffer(master, size);
    icoPngs.push({ size, buf });
    console.log(`  ICO layer: ${size}x${size} (${(buf.length / 1024).toFixed(1)} KB)`);
  }
  const ico = buildICO(icoPngs);
  writeFileSync(join(ICONS_DIR, 'icon.ico'), ico);
  console.log(`  OK: icon.ico (${(ico.length / 1024).toFixed(0)} KB, ${icoSizes.length} layers)`);
  writeFileSync(join(PUBLIC_DIR, 'favicon.ico'), ico);
  console.log(`  OK: public/favicon.ico`);

  try {
    const iconsetDir = join(ICONS_DIR, 'icon.iconset');
    mkdirSync(iconsetDir, { recursive: true });

    const icnsSizes = [
      { size: 16, name: 'icon_16x16.png' },
      { size: 32, name: 'icon_16x16@2x.png' },
      { size: 32, name: 'icon_32x32.png' },
      { size: 64, name: 'icon_32x32@2x.png' },
      { size: 128, name: 'icon_128x128.png' },
      { size: 256, name: 'icon_128x128@2x.png' },
      { size: 256, name: 'icon_256x256.png' },
      { size: 512, name: 'icon_256x256@2x.png' },
      { size: 512, name: 'icon_512x512.png' },
      { size: 1024, name: 'icon_512x512@2x.png' },
    ];

    console.log('');
    for (const { size, name } of icnsSizes) {
      await generateRoundPNG(master, size, join(iconsetDir, name));
    }

    execSync(`iconutil -c icns "${iconsetDir}" -o "${join(ICONS_DIR, 'icon.icns')}"`, { stdio: 'pipe' });
    execSync(`rm -rf "${iconsetDir}"`);
    console.log('  OK: icon.icns');
  } catch (err) {
    console.warn('  WARN: Could not generate .icns:', err.message);
  }

  console.log('\n[generateIcons] Done. All icons are high-quality circular.');
}

// ─── ICO Builder ─────────────────────────────────────────────────────────

function buildICO(images) {
  const headerSize = 6 + images.length * 16;
  const header = Buffer.alloc(headerSize);

  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(images.length, 4);

  let offset = headerSize;
  const chunks = [header];

  for (let i = 0; i < images.length; i++) {
    const { size, buf } = images[i];
    const entryOffset = 6 + i * 16;

    header.writeUInt8(size < 256 ? size : 0, entryOffset);
    header.writeUInt8(size < 256 ? size : 0, entryOffset + 1);
    header.writeUInt8(0, entryOffset + 2);
    header.writeUInt8(0, entryOffset + 3);
    header.writeUInt16LE(1, entryOffset + 4);
    header.writeUInt16LE(32, entryOffset + 6);
    header.writeUInt32LE(buf.length, entryOffset + 8);
    header.writeUInt32LE(offset, entryOffset + 12);

    chunks.push(buf);
    offset += buf.length;
  }

  return Buffer.concat(chunks);
}

main().catch(err => {
  console.error('[generateIcons] Error:', err);
  process.exit(1);
});
