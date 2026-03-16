import { NextRequest, NextResponse } from 'next/server';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as db from '@/lib/database';
import { checkLlamaConnection } from '@/lib/ai/llm';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const key = searchParams.get('key');

  // GET /api/settings?key=note-order → return specific setting
  if (key) {
    const value = db.getSetting(key);
    return NextResponse.json({ key, value: value ? JSON.parse(value) : null });
  }

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
    // Write theme file for Rust splash screen to read at next startup
    try {
      const themeDir = path.join(os.homedir(), '.epito');
      fs.mkdirSync(themeDir, { recursive: true });
      fs.writeFileSync(path.join(themeDir, 'theme'), body.theme);
    } catch {}
  }

  // PUT /api/settings with { key: "note-order", value: [...] }
  if (body.key !== undefined && body.value !== undefined) {
    db.setSetting(body.key, JSON.stringify(body.value));
    return NextResponse.json({ key: body.key, value: body.value });
  }

  const llama = await checkLlamaConnection();

  return NextResponse.json({
    theme: db.getSetting('theme') || 'dark',
    llmConnected: llama.connected,
    llmModel: llama.currentModel,
  });
}
