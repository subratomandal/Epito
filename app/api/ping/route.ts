import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// Zero-dependency readiness probe. Imports NOTHING — no database,
// no native modules. Even if better-sqlite3/sharp/etc fail to load,
// this endpoint still returns 200.
// Used by Tauri to know the HTTP server is alive before navigating.
export async function GET() {
  return NextResponse.json({ status: 'ok' });
}
