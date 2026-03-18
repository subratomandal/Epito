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
  const attachment = db.getAttachment(id);
  if (!attachment) return NextResponse.json({ error: 'Not found' }, { status: 404 });
  return NextResponse.json(attachment);
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const attachment = db.getAttachment(id);
  db.deleteAttachment(id);

  if (attachment?.file_path) {
    const filename = path.basename(attachment.file_path);
    if (filename) {
      unlink(path.join(UPLOAD_DIR, filename)).catch(() => {});
    }
  }

  return NextResponse.json({ ok: true });
}
