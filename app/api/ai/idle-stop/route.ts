import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// Signal llama-server to stop and release memory after idle timeout.
export async function POST() {
  try {
    const llamaPort = process.env.LLAMA_SERVER_PORT || '8080';

    // Unload model to free VRAM
    await fetch(`http://127.0.0.1:${llamaPort}/slots/0?action=erase`, {
      method: 'POST',
      signal: AbortSignal.timeout(3000),
    }).catch(() => {});

    const fs = await import('fs');
    const path = await import('path');
    const signalDir = process.env.EPITO_DATA_DIR || path.resolve(process.cwd(), 'data');
    fs.mkdirSync(signalDir, { recursive: true });
    fs.writeFileSync(path.join(signalDir, '.idle-stop'), Date.now().toString());

    return NextResponse.json({ status: 'idle_stop_signaled' });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
