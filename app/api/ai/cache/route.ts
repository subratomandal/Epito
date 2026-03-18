import { NextRequest, NextResponse } from 'next/server';
import { createHash } from 'crypto';
import {
  getChunkCache, upsertChunkCache, updateChunkCacheSummary,
  updateChunkCacheExplanation, clearChunkCache, pruneStaleChunkCache,
} from '@/notes/database';
import { chunkByWords, chunkForExplain, cleanInputText } from '@/model/llm';

export const dynamic = 'force-dynamic';

function hashText(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 16);
}

export async function GET(request: NextRequest) {
  const sourceId = request.nextUrl.searchParams.get('sourceId');
  if (!sourceId) {
    return NextResponse.json({ error: 'sourceId required' }, { status: 400 });
  }

  const cache = getChunkCache(sourceId);
  return NextResponse.json({ chunks: cache });
}

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }
  const { action, sourceId, text } = body;

  if (!action) {
    return NextResponse.json({ error: 'action required' }, { status: 400 });
  }

  if (action === 'prepare-summary') {
    if (!sourceId || !text) {
      return NextResponse.json({ error: 'sourceId and text required' }, { status: 400 });
    }
    const cleaned = cleanInputText(text);
    const sections = chunkByWords(cleaned, 150);
    return prepareWithCache(`${sourceId}:summary`, sections, 'summary');
  }

  if (action === 'prepare-explain') {
    if (!sourceId || !text) {
      return NextResponse.json({ error: 'sourceId and text required' }, { status: 400 });
    }
    const sections = chunkForExplain(text);
    return prepareWithCache(`${sourceId}:explain`, sections, 'explain');
  }

  if (action === 'clear') {
    if (!sourceId) {
      return NextResponse.json({ error: 'sourceId required' }, { status: 400 });
    }
    clearChunkCache(`${sourceId}:summary`);
    clearChunkCache(`${sourceId}:explain`);
    return NextResponse.json({ ok: true });
  }

  return NextResponse.json({ error: `Unknown action: ${action}` }, { status: 400 });
}

function prepareWithCache(sourceId: string, sections: string[], mode: 'summary' | 'explain') {
  pruneStaleChunkCache(sourceId, sections.length);

  const result = sections.map((text, index) => {
    const hash = hashText(text);
    const cacheId = upsertChunkCache(sourceId, index, text, hash);
    return { cacheId, index, text, hash };
  });

  const cache = getChunkCache(sourceId);
  const cacheMap = new Map(cache.map(c => [c.chunk_index, c]));

  const chunks = result.map(r => {
    const cached = cacheMap.get(r.index);
    return {
      cacheId: r.cacheId,
      index: r.index,
      text: r.text,
      summary: cached?.summary || null,
      explanation: cached?.explanation || null,
      cached: mode === 'summary' ? !!cached?.summary : !!cached?.explanation,
    };
  });

  const totalSections = chunks.length;
  const cachedCount = chunks.filter(c => c.cached).length;

  return NextResponse.json({ totalSections, cachedCount, chunks });
}

export async function PUT(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }
  const { chunkId, type, result } = body;

  if (!chunkId || !type || !result) {
    return NextResponse.json({ error: 'chunkId, type, and result required' }, { status: 400 });
  }

  if (type === 'summary') {
    updateChunkCacheSummary(chunkId, result);
  } else if (type === 'explanation') {
    updateChunkCacheExplanation(chunkId, result);
  } else {
    return NextResponse.json({ error: `Unknown type: ${type}` }, { status: 400 });
  }

  return NextResponse.json({ ok: true });
}
