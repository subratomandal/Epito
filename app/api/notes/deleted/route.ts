import { NextResponse } from 'next/server';
import * as db from '@/lib/database';

export const dynamic = 'force-dynamic';

export async function GET() {
  db.purgeOldDeleted(7);
  const deleted = db.getDeletedNotes();
  return NextResponse.json(deleted);
}
