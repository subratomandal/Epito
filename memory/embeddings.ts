import path from 'path';

let embedder: unknown = null;
let initPromise: Promise<void> | null = null;

export const EMBEDDING_DIM = 384;

export async function initEmbeddings(): Promise<void> {
  if (embedder) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const { pipeline, env } = await import('@xenova/transformers') as {
      pipeline: (task: string, model: string, opts?: Record<string, unknown>) => Promise<unknown>;
      env: { cacheDir: string; allowLocalModels: boolean; allowRemoteModels: boolean };
    };

    env.cacheDir = process.env.EPITO_DATA_DIR ? path.resolve(process.env.EPITO_DATA_DIR, 'models') : path.resolve(process.cwd(), 'data', 'models');
    env.allowLocalModels = true;
    env.allowRemoteModels = true;

    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      quantized: true,
    });

    console.log('[AI] Embedding model loaded');
  })();

  return initPromise;
}

export async function generateEmbedding(text: string): Promise<number[]> {
  await initEmbeddings();

  // all-MiniLM-L6-v2 handles ~256 word-pieces (~1500 chars). 2000 chars is safe;
  // tail content beyond that is covered by the 25% overlap in chunking.
  const truncated = text.slice(0, 2000);
  const result = await (embedder as (text: string, opts: Record<string, unknown>) => Promise<{ data: Float32Array }>)(
    truncated,
    { pooling: 'mean', normalize: true }
  );

  return Array.from(result.data);
}

export function isReady(): boolean {
  return embedder !== null;
}
