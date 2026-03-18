import { spawn } from 'child_process';
import { mkdir } from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import os from 'os';

if (typeof globalThis.DOMMatrix === 'undefined') {
  (globalThis as Record<string, unknown>).DOMMatrix = class DOMMatrix {
    m11=1;m12=0;m13=0;m14=0;m21=0;m22=1;m23=0;m24=0;
    m31=0;m32=0;m33=1;m34=0;m41=0;m42=0;m43=0;m44=1;
    a=1;b=0;c=0;d=1;e=0;f=0;is2D=true;isIdentity=true;
    constructor(init?: number[] | string) {
      if (Array.isArray(init)) {
        const v = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
        for (let i = 0; i < init.length; i++) v[i] = init[i];
        [this.m11,this.m12,this.m13,this.m14,this.m21,this.m22,this.m23,this.m24,
         this.m31,this.m32,this.m33,this.m34,this.m41,this.m42,this.m43,this.m44] = v;
        this.a=this.m11;this.b=this.m12;this.c=this.m21;this.d=this.m22;this.e=this.m41;this.f=this.m42;
        this.isIdentity=false;
      }
    }
    inverse() { return new DOMMatrix(); }
    multiply() { return new DOMMatrix(); }
    scale() { return new DOMMatrix(); }
    translate() { return new DOMMatrix(); }
    transformPoint(p?: {x:number;y:number}) { return p || {x:0,y:0}; }
    static fromMatrix() { return new DOMMatrix(); }
    static fromFloat32Array(a: Float32Array) { return new DOMMatrix(Array.from(a)); }
    static fromFloat64Array(a: Float64Array) { return new DOMMatrix(Array.from(a)); }
  };
}
if (typeof globalThis.ImageData === 'undefined') {
  (globalThis as Record<string, unknown>).ImageData = class ImageData {
    width: number; height: number; data: Uint8ClampedArray;
    constructor(w: number, h: number) {
      this.width = w || 0; this.height = h || 0;
      this.data = new Uint8ClampedArray(this.width * this.height * 4);
    }
  };
}
if (typeof globalThis.Path2D === 'undefined') {
  (globalThis as Record<string, unknown>).Path2D = class Path2D {
    addPath() {} closePath() {} moveTo() {} lineTo() {}
    bezierCurveTo() {} quadraticCurveTo() {} arc() {} arcTo() {} ellipse() {} rect() {}
  };
}

const DATA_DIR = process.env.EPITO_DATA_DIR || path.resolve(process.cwd(), 'data');

const PYTHON_CMD = os.platform() === 'win32' ? 'python' : 'python3';

export interface OCRResult {
  text: string;
  pageCount: number;
  engine: 'paddleocr' | 'tesseract' | 'pdf-parse' | 'mammoth';
  pages?: { page: number; text: string }[];
}

let _paddleAvailable: boolean | null = null;

async function isPaddleOCRAvailable(): Promise<boolean> {
  if (_paddleAvailable !== null) return _paddleAvailable;

  return new Promise((resolve) => {
    const proc = spawn(PYTHON_CMD, ['-c', 'from paddleocr import PaddleOCR; print("ok")']);
    let out = '';
    proc.stdout.on('data', (d: Buffer) => { out += d.toString(); });
    proc.on('close', (code) => {
      _paddleAvailable = code === 0 && out.trim() === 'ok';
      console.log(`[OCR] PaddleOCR ${_paddleAvailable ? 'available' : 'not available — using Tesseract fallback'}`);
      resolve(_paddleAvailable);
    });
    proc.on('error', () => {
      _paddleAvailable = false;
      resolve(false);
    });
    setTimeout(() => { proc.kill(); _paddleAvailable = false; resolve(false); }, 15000);
  });
}

export function getOCREngine(): string {
  if (_paddleAvailable === true) return 'paddleocr';
  return 'tesseract';
}

export async function preprocessImage(buffer: Buffer): Promise<{
  buffer: Buffer;
  width: number;
  height: number;
}> {
  const sharp = (await import('sharp')).default;
  const metadata = await sharp(buffer).metadata();
  const width = metadata.width || 0;
  const height = metadata.height || 0;

  const processed = await sharp(buffer)
    .grayscale()
    .normalize()
    .sharpen({ sigma: 1.0 })
    .resize({ width: Math.min(width, 3000), withoutEnlargement: true })
    .png()
    .toBuffer();

  return { buffer: processed, width, height };
}

export function computeFileHash(buffer: Buffer): string {
  return crypto.createHash('sha256').update(buffer).digest('hex').slice(0, 16);
}

const ocrCache = new Map<string, OCRResult>();

function getCached(hash: string): OCRResult | null {
  return ocrCache.get(hash) || null;
}

function setCache(hash: string, result: OCRResult): void {
  if (ocrCache.size > 100) {
    const first = ocrCache.keys().next().value;
    if (first) ocrCache.delete(first);
  }
  ocrCache.set(hash, result);
}

async function runPaddleOCR(filePath: string, type: 'image' | 'pdf'): Promise<OCRResult> {
  const scriptPath = path.join(process.cwd(), 'scripts', 'paddleOcr.py');

  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_CMD, [scriptPath, type, filePath]);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on('data', (d: Buffer) => { stderr += d.toString(); });

    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`PaddleOCR exited ${code}: ${stderr.slice(0, 500)}`));
        return;
      }
      try {
        const data = JSON.parse(stdout);
        if (!data.success) {
          reject(new Error(data.error || 'PaddleOCR failed'));
          return;
        }
        resolve({
          text: data.text || '',
          pageCount: data.page_count || 1,
          engine: 'paddleocr',
          pages: data.pages,
        });
      } catch {
        reject(new Error('Failed to parse PaddleOCR output'));
      }
    });

    proc.on('error', reject);

    const timer = setTimeout(() => { proc.kill(); reject(new Error('PaddleOCR timed out')); }, 300000);
    proc.on('close', () => clearTimeout(timer));
  });
}

async function runTesseract(buffer: Buffer): Promise<OCRResult> {
  const Tesseract = await import('tesseract.js');

  const tessDataDir = path.join(DATA_DIR, 'tessdata');
  await mkdir(tessDataDir, { recursive: true });

  const workerPromise = Tesseract.createWorker('eng', 1, {
    cachePath: tessDataDir,
  });

  const timeoutPromise = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error('Tesseract worker creation timed out (120s)')), 120000)
  );

  const worker = await Promise.race([workerPromise, timeoutPromise]);
  try {
    const { data: { text } } = await worker.recognize(buffer);
    return {
      text: cleanOCRText(text),
      pageCount: 1,
      engine: 'tesseract',
    };
  } finally {
    await worker.terminate();
  }
}

async function extractPDFTextLayer(buffer: Buffer): Promise<{ text: string; pageCount: number } | null> {
  try {
    const pdfModule = await import('pdf-parse') as Record<string, unknown>;

    let text = '';
    let pageCount = 0;

    if ('PDFParse' in pdfModule && typeof pdfModule.PDFParse === 'function') {
      const PDFParse = pdfModule.PDFParse as new (opts: { data: Buffer }) => {
        getText(): Promise<{ text: string; total: number }>;
        destroy(): Promise<void>;
      };
      const parser = new PDFParse({ data: buffer });
      const result = await parser.getText();
      text = result.text || '';
      pageCount = result.total || 0;
      await parser.destroy();
    } else {
      const pdfParse = (pdfModule as { default?: (buf: Buffer) => Promise<{ text: string; numpages: number }> }).default || pdfModule;
      if (typeof pdfParse === 'function') {
        const result = await (pdfParse as (buf: Buffer) => Promise<{ text: string; numpages: number }>)(buffer);
        text = result.text || '';
        pageCount = result.numpages || 0;
      }
    }

    const stripped = text.replace(/\s+/g, '').trim();
    if (stripped.length < 50) {
      return null;
    }

    return { text, pageCount };
  } catch {
    return null;
  }
}

async function extractDOCXText(buffer: Buffer): Promise<{ text: string; pageCount: number }> {
  const mammoth = await import('mammoth');
  const result = await mammoth.extractRawText({ buffer });
  const text = result.value;
  return { text, pageCount: Math.ceil(text.length / 3000) };
}

export function cleanOCRText(text: string): string {
  return text
    .replace(/(\w)-\n(\w)/g, '$1$2')
    .replace(/[^\S\n]+/g, ' ')
    .replace(/^ +| +$/gm, '')
    .replace(/^[|~`^\\{}]+$/gm, '')
    .replace(/^[^\w\s]{1,3}$/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[\u200B-\u200D\uFEFF\u00AD]/g, '')
    .trim();
}

export async function ocrImage(filePath: string, buffer: Buffer): Promise<OCRResult> {
  const hash = computeFileHash(buffer);
  const cached = getCached(hash);
  if (cached) {
    console.log(`[OCR] Cache hit: ${hash}`);
    return cached;
  }

  let result: OCRResult;
  const hasPaddle = await isPaddleOCRAvailable();

  if (hasPaddle) {
    try {
      result = await runPaddleOCR(filePath, 'image');
      result.text = cleanOCRText(result.text);
      console.log(`[OCR] PaddleOCR image: ${result.text.length} chars`);
    } catch (err) {
      console.warn('[OCR] PaddleOCR failed, falling back to Tesseract:', err);
      const { buffer: preprocessed } = await preprocessImage(buffer);
      result = await runTesseract(preprocessed);
    }
  } else {
    const { buffer: preprocessed } = await preprocessImage(buffer);
    result = await runTesseract(preprocessed);
    console.log(`[OCR] Tesseract image: ${result.text.length} chars`);
  }

  setCache(hash, result);
  return result;
}

export async function ocrPDF(filePath: string, buffer: Buffer): Promise<OCRResult> {
  const hash = computeFileHash(buffer);
  const cached = getCached(hash);
  if (cached) {
    console.log(`[OCR] Cache hit: ${hash}`);
    return cached;
  }

  const textLayer = await extractPDFTextLayer(buffer);
  if (textLayer && textLayer.text.trim().length >= 50) {
    const result: OCRResult = {
      text: cleanOCRText(textLayer.text),
      pageCount: textLayer.pageCount,
      engine: 'pdf-parse',
    };
    console.log(`[OCR] PDF text layer: ${result.text.length} chars, ${result.pageCount} pages`);
    setCache(hash, result);
    return result;
  }

  const hasPaddle = await isPaddleOCRAvailable();
  if (hasPaddle) {
    try {
      const result = await runPaddleOCR(filePath, 'pdf');
      result.text = cleanOCRText(result.text);
      console.log(`[OCR] PaddleOCR PDF: ${result.text.length} chars, ${result.pageCount} pages`);
      setCache(hash, result);
      return result;
    } catch (err) {
      console.warn('[OCR] PaddleOCR PDF failed:', err);
    }
  }

  const result: OCRResult = {
    text: textLayer?.text ? cleanOCRText(textLayer.text) : '',
    pageCount: textLayer?.pageCount || 0,
    engine: 'pdf-parse',
  };
  setCache(hash, result);
  return result;
}

export async function ocrDOCX(buffer: Buffer): Promise<OCRResult> {
  const hash = computeFileHash(buffer);
  const cached = getCached(hash);
  if (cached) return cached;

  const { text, pageCount } = await extractDOCXText(buffer);
  const result: OCRResult = {
    text: cleanOCRText(text),
    pageCount,
    engine: 'mammoth',
  };
  setCache(hash, result);
  return result;
}
