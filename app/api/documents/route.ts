import { NextResponse } from 'next/server';
import * as db from '@/lib/database';

export const dynamic = 'force-dynamic';

export async function GET() {
  const documents = db.getAllDocuments();
  return NextResponse.json(documents);
}
