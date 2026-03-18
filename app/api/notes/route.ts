import { NextRequest, NextResponse } from 'next/server';
import * as db from '@/notes/database';

export const dynamic = 'force-dynamic';

export async function GET() {
  const notes = db.getAllNotes();
  return NextResponse.json(notes);
}

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }
  const { title = 'Untitled', content = '', folder = '', tags = [] } = body;
  const id = db.createNote(title, content, '', folder, tags);
  const note = db.getNote(id);
  return NextResponse.json(note, { status: 201 });
}
