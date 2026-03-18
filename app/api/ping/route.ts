import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// Zero-dependency readiness probe. No DB or native module imports.
export async function GET() {
  return NextResponse.json({ status: 'ok' });
}
