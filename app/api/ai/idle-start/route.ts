import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// Restart llama-server after idle stop.
export async function POST() {
  try {
    const llamaPort = process.env.LLAMA_SERVER_PORT || '8080';

    try {
      const health = await fetch(`http://127.0.0.1:${llamaPort}/health`, {
        signal: AbortSignal.timeout(2000),
      });
      if (health.ok) {
        return NextResponse.json({ status: 'already_running' });
      }
    } catch {}

    const fs = await import('fs');
    const path = await import('path');
    const signalDir = process.env.EPITO_DATA_DIR || path.resolve(process.cwd(), 'data');
    fs.mkdirSync(signalDir, { recursive: true });
    fs.writeFileSync(path.join(signalDir, '.idle-start'), Date.now().toString());

    const maxWait = 120_000;
    const start = Date.now();
    while (Date.now() - start < maxWait) {
      await new Promise(r => setTimeout(r, 1000));
      try {
        const res = await fetch(`http://127.0.0.1:${llamaPort}/health`, {
          signal: AbortSignal.timeout(2000),
        });
        if (res.ok) {
          return NextResponse.json({ status: 'started' });
        }
      } catch {}
    }

    return NextResponse.json({ error: 'timeout' }, { status: 504 });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
