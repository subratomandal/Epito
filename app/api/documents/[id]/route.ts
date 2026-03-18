import { NextRequest, NextResponse } from 'next/server';
import * as db from '@/notes/database';
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
  const doc = db.getDocument(id);
  if (!doc) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  const chunks = db.getChunksByNote(id);
  return NextResponse.json({ ...doc, chunks });
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

  const fields: { tags?: string[]; plain_text?: string } = {};
  if (body.tags !== undefined) fields.tags = body.tags;
  if (body.plain_text !== undefined) fields.plain_text = body.plain_text;

  db.updateDocument(id, fields);
  const doc = db.getDocument(id);
  return NextResponse.json(doc);
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const doc = db.getDocument(id);
  db.deleteDocument(id);

  if (doc?.file_path) {
    const filename = path.basename(doc.file_path);
    if (filename) {
      unlink(path.join(UPLOAD_DIR, filename)).catch(() => {});
    }
  }

  return NextResponse.json({ ok: true });
}
