import { NextResponse } from 'next/server';
import * as db from '@/lib/database';

export const dynamic = 'force-dynamic';

export async function GET() {
  const topics = db.getAllTopics();
  return NextResponse.json(topics);
}
