import { NextResponse } from 'next/server';
import * as db from '@/lib/database';
import { installShutdownHandlers } from '@/lib/lifecycle';

export const dynamic = 'force-dynamic';

let shutdownHandlersInstalled = false;
function ensureShutdownHandlers() {
  if (!shutdownHandlersInstalled) {
    shutdownHandlersInstalled = true;
    installShutdownHandlers();
  }
}

export async function GET(request: Request) {
  ensureShutdownHandlers();

  const { searchParams } = new URL(request.url);
  const isReadinessProbe = searchParams.get('ready') !== null;

  // Lightweight readiness probe: returns 200 without touching DB/native modules.
  // Used by Tauri to know the HTTP server is up before navigating the window.
  if (isReadinessProbe) {
    return NextResponse.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
    });
  }

  try {
    const stats = db.getStats();
    return NextResponse.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      stats,
    });
  } catch (err) {
    return NextResponse.json({
      status: 'error',
      timestamp: new Date().toISOString(),
      error: err instanceof Error ? err.message : 'Unknown error',
    }, { status: 503 });
  }
}
