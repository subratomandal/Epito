import { NextRequest, NextResponse } from 'next/server';
import * as db from '@/notes/database';
import path from 'path';
import { mkdir, writeFile } from 'fs/promises';
import { randomUUID } from 'crypto';

export const dynamic = 'force-dynamic';

const DATA_DIR = process.env.EPITO_DATA_DIR || path.resolve(process.cwd(), 'data');
const UPLOAD_DIR = path.resolve(DATA_DIR, 'uploads');
const MAX_ATTACHMENT_SIZE = 20 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set(['pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt', 'md']);

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File | null;
    const noteId = formData.get('noteId') as string | null;

    if (!file || !noteId) {
      return NextResponse.json({ error: 'File and noteId are required' }, { status: 400 });
    }

    if (file.size > MAX_ATTACHMENT_SIZE) {
      return NextResponse.json({ error: 'File too large. Max 20MB.' }, { status: 400 });
    }

    const ext = path.extname(file.name).toLowerCase().replace('.', '');
    if (!ALLOWED_EXTENSIONS.has(ext)) {
      return NextResponse.json({ error: 'Unsupported file type' }, { status: 400 });
    }

    const note = db.getNote(noteId);
    if (!note) {
      return NextResponse.json({ error: 'Note not found' }, { status: 404 });
    }

    await mkdir(UPLOAD_DIR, { recursive: true });
    const safeName = `${randomUUID()}.${ext}`;
    const filePath = path.join(UPLOAD_DIR, safeName);
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(filePath, buffer);

    const mimeType = file.type || 'application/octet-stream';
    let fileType = 'other';
    if (mimeType === 'application/pdf' || ext === 'pdf') fileType = 'pdf';
    else if (mimeType.startsWith('image/')) fileType = 'image';

    const id = db.createAttachment(noteId, file.name, `/api/upload/${safeName}`, fileType, buffer.length, mimeType);

    const attachment = db.getAttachment(id);
    return NextResponse.json(attachment, { status: 201 });
  } catch (err) {
    console.error('[Attachments] Upload error:', err);
    return NextResponse.json({ error: 'Attachment upload failed' }, { status: 500 });
  }
}
