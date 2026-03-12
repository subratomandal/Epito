import { NextRequest, NextResponse } from 'next/server';
import * as db from '@/lib/database';
import { unlink } from 'fs/promises';
import path from 'path';

export const dynamic = 'force-dynamic';

const DATA_DIR = process.env.EPITO_DATA_DIR || path.resolve(process.cwd(), 'data');
const UPLOAD_DIR = path.resolve(DATA_DIR, 'uploads');

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const img = db.getImage(id);
  if (!img) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  const chunks = db.getChunksByNote(id);
  return NextResponse.json({ ...img, chunks });
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  const fields: { tags?: string[]; ocr_text?: string } = {};
  if (body.tags !== undefined) fields.tags = body.tags;
  if (body.ocr_text !== undefined) fields.ocr_text = body.ocr_text;

  db.updateImage(id, fields);
  const img = db.getImage(id);
  return NextResponse.json(img);
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const img = db.getImage(id);
  db.deleteImage(id);

  if (img?.file_path) {
    const filename = path.basename(img.file_path);
    if (filename) {
      unlink(path.join(UPLOAD_DIR, filename)).catch(() => {});
    }
  }

  return NextResponse.json({ ok: true });
}
