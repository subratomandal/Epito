import { NextResponse } from 'next/server';
import { getModelStatus, checkLlamaConnection } from '@/lib/ai/llm';
import { isReady as isEmbeddingReady } from '@/lib/ai/embeddings';
import { getOCREngine } from '@/lib/ocr';

export const dynamic = 'force-dynamic';

export async function GET() {
  const llm = getModelStatus();
  const llama = await checkLlamaConnection();

  return NextResponse.json({
    llm: {
      ...llm,
      available: llama.connected,
      loaded: llama.connected,
    },
    llm_server: {
      connected: llama.connected,
      models: llama.models,
      currentModel: llama.currentModel,
    },
    embeddings: {
      ready: isEmbeddingReady(),
      model: 'all-MiniLM-L6-v2',
    },
    ocr: {
      engine: getOCREngine(),
    },
  });
}
