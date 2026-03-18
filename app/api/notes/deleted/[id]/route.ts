import { NextRequest, NextResponse } from 'next/server';
import * as db from '@/notes/database';

export const dynamic = 'force-dynamic';

export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  db.restoreNote(id);
  return NextResponse.json({ ok: true });
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  db.permanentDeleteNote(id);
  return NextResponse.json({ ok: true });
}
