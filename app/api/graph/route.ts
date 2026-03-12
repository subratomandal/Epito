import { NextResponse } from 'next/server';
import * as db from '@/lib/database';

export const dynamic = 'force-dynamic';

export async function GET() {
  const topics = db.getAllTopics();
  const links = db.getAllLinks();
  const stats = db.getStats();
  return NextResponse.json({ topics, links, stats });
}
