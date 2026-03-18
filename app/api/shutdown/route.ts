import { NextResponse } from 'next/server';
import { closeDb } from '@/notes/database';

export const dynamic = 'force-dynamic';

// Graceful shutdown: flush WAL and close database before process exit.
export async function POST(request: Request) {
  const origin = request.headers.get('origin') || '';
  const referer = request.headers.get('referer') || '';
  const isLocalOrigin =
    !origin || // server-side fetch (no origin)
    origin.startsWith('http://127.0.0.1') ||
    origin.startsWith('http://localhost') ||
    origin.startsWith('tauri://');
  const isLocalReferer =
    !referer ||
    referer.startsWith('http://127.0.0.1') ||
    referer.startsWith('http://localhost') ||
    referer.startsWith('tauri://');

  if (!isLocalOrigin || !isLocalReferer) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  try {
    closeDb();
  } catch (err) {
    console.error('[Shutdown] Error closing database:', err);
  }

  const response = NextResponse.json({ status: 'shutting_down' });
  setTimeout(() => process.exit(0), 100);

  return response;
}
