import { NextRequest, NextResponse } from 'next/server';
import * as db from '@/notes/database';
import { processNote } from '@/inference/pipeline';
import { stripHtml } from '@/common/utils';

export const dynamic = 'force-dynamic';

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const note = db.getNote(id);
  if (!note) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  const chunks = db.getChunksByNote(id);
  const attachments = db.getAttachmentsByNote(id);
  return NextResponse.json({ ...note, chunks, attachments });
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

  const fields: Record<string, unknown> = {};
  if (body.title !== undefined) fields.title = body.title;
  if (body.content !== undefined) {
    fields.content = body.content;
    fields.plain_text = stripHtml(body.content);
  }
  if (body.folder !== undefined) fields.folder = body.folder;
  if (body.tags !== undefined) fields.tags = body.tags;

  db.updateNote(id, fields as Parameters<typeof db.updateNote>[1]);

  if (body.content !== undefined) {
    processNote(id).catch(err => console.error('[AI] Process error:', err.message));
  }

  const note = db.getNote(id);
  return NextResponse.json(note);
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  db.deleteNote(id);
  return NextResponse.json({ ok: true });
}
