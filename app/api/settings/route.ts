import { NextRequest, NextResponse } from 'next/server';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as db from '@/notes/database';
import { checkLlamaConnection } from '@/model/llm';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const key = searchParams.get('key');

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
    try {
      const themeDir = path.join(os.homedir(), '.epito');
      fs.mkdirSync(themeDir, { recursive: true });
      fs.writeFileSync(path.join(themeDir, 'theme'), body.theme);
    } catch {}
  }

  const ALLOWED_SETTING_KEYS = new Set(['note-order', 'sidebar-collapsed', 'insight-collapsed']);
  if (body.key !== undefined && body.value !== undefined) {
    if (typeof body.key !== 'string' || !ALLOWED_SETTING_KEYS.has(body.key)) {
      return NextResponse.json({ error: 'Invalid setting key' }, { status: 400 });
    }
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
