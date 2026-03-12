import { NextRequest, NextResponse } from 'next/server';
import { findRelatedNotes } from '@/lib/ai/pipeline';

export const dynamic = 'force-dynamic';

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const related = await findRelatedNotes(id);
    return NextResponse.json(related);
  } catch (err) {
    console.error('[API] Related notes error:', err);
    return NextResponse.json([]);
  }
}
