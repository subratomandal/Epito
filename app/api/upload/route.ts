import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';
import { randomUUID } from 'crypto';
import * as db from '@/notes/database';
import { processDocument, processImage } from '@/inference/pipeline';
import { ocrImage, ocrPDF, ocrDOCX, preprocessImage } from '@/inference/ocr';

export const dynamic = 'force-dynamic';

const DATA_DIR = process.env.EPITO_DATA_DIR || path.resolve(process.cwd(), 'data');
const UPLOAD_DIR = path.resolve(DATA_DIR, 'uploads');
const MAX_SIZE = 50 * 1024 * 1024;

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File | null;
    const uploadType = formData.get('type') as string | null;

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    if (file.size > MAX_SIZE) {
      return NextResponse.json({ error: 'File too large. Max 50MB.' }, { status: 400 });
    }

    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    const DOC_EXTENSIONS = new Set(['pdf', 'docx']);
    const IMG_EXTENSIONS = new Set(['png', 'jpg', 'jpeg']);

    if (uploadType === 'document') {
      if (!DOC_EXTENSIONS.has(ext)) {
        return NextResponse.json(
          { error: 'Only PDF and DOCX files can be uploaded as documents.' },
          { status: 400 }
        );
      }
    } else if (uploadType === 'image') {
      if (!IMG_EXTENSIONS.has(ext)) {
        return NextResponse.json(
          { error: 'Only PNG, JPG, and JPEG files can be uploaded as images.' },
          { status: 400 }
        );
      }
    } else if (!DOC_EXTENSIONS.has(ext) && !IMG_EXTENSIONS.has(ext)) {
      return NextResponse.json(
        { error: 'Unsupported file type. Supported: PDF, DOCX, PNG, JPG, JPEG' },
        { status: 400 }
      );
    }

    const isDoc = DOC_EXTENSIONS.has(ext);
    const isImg = IMG_EXTENSIONS.has(ext);

    const buffer = Buffer.from(await file.arrayBuffer());
    await mkdir(UPLOAD_DIR, { recursive: true });

    const storedName = `${randomUUID()}.${ext}`;
    const filepath = path.join(UPLOAD_DIR, storedName);
    await writeFile(filepath, buffer);

    if (isDoc) {
      return await handleDocument(file, buffer, filepath, storedName, ext);
    } else {
      return await handleImage(file, buffer, filepath, storedName);
    }
  } catch (err) {
    console.error('[Upload] Error:', err);
    return NextResponse.json({ error: 'Upload failed' }, { status: 500 });
  }
}

async function handleDocument(
  file: File, buffer: Buffer, filepath: string, storedName: string, ext: string
) {
  const docId = db.createDocument({
    filename: file.name,
    fileType: ext === 'docx' ? 'docx' : 'pdf',
    filePath: `/api/upload/${storedName}`,
    fileSize: file.size,
    plainText: '',
    pageCount: 0,
  });

  (async () => {
    try {
      let text = '';
      let pageCount = 0;

      if (ext === 'pdf') {
        console.log(`[Upload] Starting PDF extraction: ${file.name} (${buffer.length} bytes)`);
        const result = await ocrPDF(filepath, buffer);
        text = result.text;
        pageCount = result.pageCount;
        console.log(`[Upload] PDF processed: ${file.name} (${result.engine}, ${pageCount} pages, ${text.length} chars)`);
      } else if (ext === 'docx') {
        console.log(`[Upload] Starting DOCX extraction: ${file.name} (${buffer.length} bytes)`);
        const result = await ocrDOCX(buffer);
        text = result.text;
        pageCount = result.pageCount;
        console.log(`[Upload] DOCX processed: ${file.name} (${pageCount} pages, ${text.length} chars)`);
      }

      db.updateDocument(docId, { plain_text: text, page_count: pageCount });

      try {
        await processDocument(docId);
      } catch (pipelineErr) {
        console.error('[Upload] Document pipeline error:', pipelineErr);
        db.updateDocument(docId, { status: 'error' });
      }
    } catch (err) {
      console.error('[Upload] Document extraction error:', err);
      db.updateDocument(docId, { status: 'error' });
    }
  })().catch(err => console.error('[Upload] Unhandled document error:', err));

  const doc = db.getDocument(docId);
  return NextResponse.json({ type: 'document', document: doc }, { status: 201 });
}

async function handleImage(
  file: File, buffer: Buffer, filepath: string, storedName: string
) {
  let width = 0, height = 0;
  try {
    const result = await preprocessImage(buffer);
    width = result.width;
    height = result.height;
  } catch (err) {
    console.error('[Upload] Preprocessing error:', err);
  }

  const imageId = db.createImage({
    filename: file.name,
    filePath: `/api/upload/${storedName}`,
    fileSize: file.size,
    mimeType: file.type,
    width,
    height,
  });

  (async () => {
    try {
      console.log(`[Upload] Starting image OCR: ${file.name} (${buffer.length} bytes)`);
      const result = await ocrImage(filepath, buffer);

      db.updateImage(imageId, { ocr_text: result.text });
      console.log(`[Upload] Image OCR: ${file.name} (${result.engine}, ${result.text.length} chars)`);

      try {
        await processImage(imageId);
      } catch (pipelineErr) {
        console.error('[Upload] Image pipeline error:', pipelineErr);
        db.updateImage(imageId, { status: 'error' });
      }
    } catch (err) {
      console.error('[Upload] OCR error:', err);
      db.updateImage(imageId, { status: 'error' });
    }
  })().catch(err => console.error('[Upload] Unhandled image error:', err));

  const img = db.getImage(imageId);
  return NextResponse.json({ type: 'image', image: img }, { status: 201 });
}
