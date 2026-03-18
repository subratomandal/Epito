// --- Explain Pipeline
// Splits notes into 100-word chunks, explains each one sequentially via LLM.
// Isolated from chat and summary pipelines.

import { cleanInputText, ensureLlamaRunning, holdServer, releaseServer } from './llm';

export type ExplainEvent =
  | { type: 'progress'; message: string }
  | { type: 'segmentStream'; text: string; index: number; total: number }
  | { type: 'segmentDone'; explanation: string; index: number; total: number };

const LLAMA_PORT = process.env.LLAMA_SERVER_PORT || '8080';
const LLAMA_URL = `http://127.0.0.1:${LLAMA_PORT}`;

const EXPLAIN_SYSTEM = `You are an AI assistant designed to help users understand text clearly. You will be given a short piece of text (about 100 words). Your task is to explain it in very simple, easy-to-understand language.`;

const EXPLAIN_PROMPT = (segmentText: string) =>
`Read the following text and explain it clearly.

IMPORTANT RULES:
- Do NOT summarize
- Do NOT shorten aggressively
- Do NOT repeat sentences from the text
- Focus on understanding, not compression
- Only explain the given text

PROCESS:
1. Read the text carefully
2. Identify the main idea
3. Identify difficult or technical parts
4. Explain everything in simple, clear language
5. Add missing context if needed

STYLE:
- Use simple words
- Keep sentences short and clear
- Make it feel like teaching a beginner
- Avoid jargon

Text:
"""
${segmentText}
"""

Explanation:`;

// --- Chunking

export interface ExplainChunk {
  text: string;
  index: number;
}

export function splitInto100WordChunks(text: string): ExplainChunk[] {
  const cleaned = cleanInputText(text);
  const words = cleaned.split(/\s+/).filter(w => w.length > 0);
  const chunks: ExplainChunk[] = [];

  for (let i = 0; i < words.length; i += 100) {
    chunks.push({
      text: words.slice(i, i + 100).join(' '),
      index: chunks.length,
    });
  }

  return chunks;
}

// --- Single Chunk Explanation (streaming)

export async function* explainSingleChunk(
  chunkText: string,
  index: number,
  total: number,
): AsyncGenerator<ExplainEvent> {
  await ensureLlamaRunning();
  holdServer();

  yield { type: 'progress', message: `Explaining section ${index + 1} of ${total}...` };

  try {
    const body = {
      messages: [
        { role: 'system', content: EXPLAIN_SYSTEM },
        { role: 'user', content: EXPLAIN_PROMPT(chunkText) },
      ],
      temperature: 0.7,
      top_p: 0.9,
      repeat_penalty: 1.1,
      max_tokens: 400,
      stream: true,
    };

    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`LLM error: ${res.status}`);

    const reader = res.body?.getReader();
    if (!reader) throw new Error('No body');
    const decoder = new TextDecoder();
    let accumulated = '', buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const t = line.trim();
        if (!t || !t.startsWith('data: ')) continue;
        const p = t.slice(6);
        if (p === '[DONE]') continue;
        try {
          const d = JSON.parse(p);
          const c = d.choices?.[0]?.delta?.content;
          if (c) {
            accumulated += c;
            yield { type: 'segmentStream', text: accumulated, index, total };
          }
        } catch {}
      }
    }

    accumulated = accumulated
      .replace(/<\/s>/g, '')
      .replace(/<\|im_end\|>/g, '')
      .replace(/\s*\bterminated\b\.?\s*$/i, '')
      .trim();

    yield { type: 'segmentDone', explanation: accumulated, index, total };
  } finally {
    releaseServer();
  }
}
