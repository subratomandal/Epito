import { NextResponse } from 'next/server';
import * as db from '@/notes/database';
import { installShutdownHandlers } from '@/inference/lifecycle';

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

  // Readiness probe: returns 200 without touching DB
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
