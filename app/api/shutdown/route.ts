import { NextResponse } from 'next/server';
import { closeDb } from '@/lib/database';

export const dynamic = 'force-dynamic';

// Called by Tauri before force-killing the Node.js process.
// Ensures the database is cleanly closed (WAL flushed, locks released).
// Only accepts requests from localhost.
export async function POST(request: Request) {
  const host = request.headers.get('host') || '';
  if (!host.startsWith('127.0.0.1') && !host.startsWith('localhost')) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  try {
    closeDb();
  } catch (err) {
    console.error('[Shutdown] Error closing database:', err);
  }

  // Respond before exiting so the caller gets the 200
  const response = NextResponse.json({ status: 'shutting_down' });

  // Schedule exit after response is sent
  setTimeout(() => process.exit(0), 100);

  return response;
}
