import { NextRequest, NextResponse } from 'next/server';
import * as db from '@/lib/database';
import { checkLlamaConnection } from '@/lib/ai/llm';

export const dynamic = 'force-dynamic';

export async function GET() {
  const llama = await checkLlamaConnection();
  return NextResponse.json({
    theme: db.getSetting('theme') || 'dark',
    llmConnected: llama.connected,
    llmModel: llama.currentModel,
  });
}

export async function PUT(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  if (body.theme !== undefined) {
    if (body.theme !== 'light' && body.theme !== 'dark') {
      return NextResponse.json({ error: 'Invalid theme value' }, { status: 400 });
    }
    db.setSetting('theme', body.theme);
  }

  const llama = await checkLlamaConnection();

  return NextResponse.json({
    theme: db.getSetting('theme') || 'dark',
    llmConnected: llama.connected,
    llmModel: llama.currentModel,
  });
}
