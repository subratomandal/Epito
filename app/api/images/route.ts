import { NextResponse } from 'next/server';
import * as db from '@/notes/database';

export const dynamic = 'force-dynamic';

export async function GET() {
  const images = db.getAllImages();
  return NextResponse.json(images);
}
