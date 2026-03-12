import { NextResponse } from 'next/server';
import * as db from '@/lib/database';

export const dynamic = 'force-dynamic';

export async function GET() {
  const tags = db.getAllTagsWithCounts();
  return NextResponse.json(tags);
}
