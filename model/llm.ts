import { contextualRetrieveForChat } from '@/inference/pipeline';
import * as db from '@/notes/database';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

const LLAMA_PORT = process.env.LLAMA_SERVER_PORT || '8080';
const LLAMA_URL = `http://127.0.0.1:${LLAMA_PORT}`;
const MODEL = 'mistral-7b-instruct';

// --- Instant Offload Architecture
// KV cache cleared instantly after inference. Process killed after idle timeout.
// Lifecycle: idle -> on-demand spawn -> inference -> KV clear -> idle timeout -> kill.
const PROCESS_KILL_TIMEOUT_MS = 7_000;    // Kill process after 7s idle

// Signal file directory (shared with Rust runtime controller)
const DATA_DIR = process.env.EPITO_DATA_DIR || path.join(os.homedir(), '.epito', 'data');

console.log(`[LLM] Runtime: url=${LLAMA_URL}, model=${MODEL}, offload=instant, kill=${PROCESS_KILL_TIMEOUT_MS / 1000}s`);

let inferenceActive = false;
const inferenceQueue: Array<{ resolve: () => void }> = [];

// --- Instant Offload Engine
// holdServer(): keep process alive. releaseServer(): KV clear + schedule kill.

let processKillTimer: ReturnType<typeof setTimeout> | null = null;
let llamaServerRunning = false;

/** Cancel pending kill timer — called BEFORE inference to keep server alive. */
export function holdServer(): void {
  if (processKillTimer) {
    clearTimeout(processKillTimer);
    processKillTimer = null;
  }
  llamaServerRunning = true;
}

/** Instant KV clear + schedule process kill — called AFTER inference completes. */
export function releaseServer(): void {
  // Tier 1: INSTANT KV cache eviction (fire-and-forget, non-blocking)
  // Frees ~200MB of attention memory immediately. Process stays alive.
  fetch(`${LLAMA_URL}/slots/0?action=erase`, {
    method: 'POST',
    signal: AbortSignal.timeout(3000),
  }).then(() => {
    console.log('[LLM] KV cache cleared instantly after inference. ~200MB freed.');
  }).catch(() => {
    // Server might already be dead — kill timer handles cleanup
  });

  // Tier 2: Schedule full process kill after idle timeout
  if (processKillTimer) clearTimeout(processKillTimer);
  processKillTimer = setTimeout(killLlamaProcessIdle, PROCESS_KILL_TIMEOUT_MS);
}

function killLlamaProcessIdle(): void {
  if (!llamaServerRunning) return;
  // Write signal file for Rust runtime controller to kill the process.
  // This releases ALL memory: model weights + KV cache + GPU VRAM + tensor arena.
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(path.join(DATA_DIR, '.idle-stop'), '', { flag: 'w' });
    llamaServerRunning = false;
    console.log('[LLM] Process kill signaled (idle 30s). Full memory reclaim.');
  } catch (e) {
    console.error('[LLM] Failed to write idle-stop signal:', e);
  }
}

export async function ensureLlamaRunning(): Promise<void> {
  // Fast path: if we believe it's running, verify with health check
  if (llamaServerRunning) {
    try {
      const res = await fetch(`${LLAMA_URL}/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return;
    } catch {}
    // Health failed — server died unexpectedly
    llamaServerRunning = false;
  }

  // Cancel any pending kill timer — we need the server NOW
  if (processKillTimer) {
    clearTimeout(processKillTimer);
    processKillTimer = null;
  }

  // Server not running — signal Rust runtime controller to spawn worker
  console.log('[LLM] Worker not running. Signaling Rust to spawn llama-server...');
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(path.join(DATA_DIR, '.idle-start'), '', { flag: 'w' });
  } catch (e) {
    console.error('[LLM] Failed to write idle-start signal:', e);
    throw new Error('Cannot signal llama-server restart');
  }

  // Aggressive polling: 100ms for first 10 attempts (1s), then 500ms
  // This shaves 400-900ms off the startup detection vs the old 500ms/1s pattern
  const maxWait = 180_000;
  const start = Date.now();
  let attempt = 0;

  while (Date.now() - start < maxWait) {
    attempt++;
    try {
      const res = await fetch(`${LLAMA_URL}/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        llamaServerRunning = true;
        console.log(`[LLM] Worker ready (${((Date.now() - start) / 1000).toFixed(1)}s, ${attempt} attempts)`);
        return;
      }
    } catch {}
    // Aggressive polling: 100ms for first 10 attempts, then 500ms
    await new Promise(r => setTimeout(r, attempt <= 10 ? 100 : 500));
  }
  throw new Error('llama-server failed to start within 180s');
}

async function acquireInferenceLock(): Promise<void> {
  if (!inferenceActive) {
    inferenceActive = true;
    return;
  }
  return new Promise<void>((resolve) => {
    inferenceQueue.push({ resolve });
  });
}

function releaseInferenceLock(): void {
  const next = inferenceQueue.shift();
  if (next) {
    next.resolve();
  } else {
    inferenceActive = false;
  }
}

const recentInferenceTimes: number[] = [];
const MAX_TRACKED_INFERENCES = 10;
let baselineInferenceMs = 0;

function recordInferenceTime(durationMs: number): void {
  recentInferenceTimes.push(durationMs);
  if (recentInferenceTimes.length > MAX_TRACKED_INFERENCES) {
    recentInferenceTimes.shift();
  }
  if (baselineInferenceMs === 0 && recentInferenceTimes.length >= 2) {
    baselineInferenceMs = recentInferenceTimes.reduce((a, b) => a + b, 0) / recentInferenceTimes.length;
    console.log(`[LLM] Inference baseline established: ${Math.round(baselineInferenceMs)}ms`);
  }
}

function computeThermalCooldown(): number {
  if (recentInferenceTimes.length < 3 || baselineInferenceMs === 0) return 0;
  const recent = recentInferenceTimes.slice(-3);
  const avgRecent = recent.reduce((a, b) => a + b, 0) / recent.length;
  const ratio = avgRecent / baselineInferenceMs;

  if (ratio > 2.5) {
    console.log(`[LLM] Thermal pressure CRITICAL (${ratio.toFixed(1)}x baseline) — 5s cooldown`);
    return 5000;
  }
  if (ratio > 1.8) {
    console.log(`[LLM] Thermal pressure HIGH (${ratio.toFixed(1)}x baseline) — 2s cooldown`);
    return 2000;
  }
  if (ratio > 1.4) {
    console.log(`[LLM] Thermal pressure MODERATE (${ratio.toFixed(1)}x baseline) — 500ms cooldown`);
    return 500;
  }
  return 0;
}

function adaptiveMaxTokens(requestedMax?: number): number | undefined {
  // Thermal adaptation disabled — it was halving the generation budget
  // under load, causing premature termination. The model should always
  // get its full token budget. Thermal management is handled by cooldown
  // delays between inferences, not by truncating output.
  return requestedMax;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function getModelStatus(): { available: boolean; loaded: boolean; modelPath: string; modelFilename: string } {
  return {
    available: true,
    loaded: true,
    modelPath: 'llama-server',
    modelFilename: MODEL,
  };
}

export async function checkLlamaConnection(): Promise<{ connected: boolean; models: string[]; currentModel: string }> {
  try {
    const res = await fetch(`${LLAMA_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) {
      console.log(`[LLM] Health check failed: status ${res.status}`);
      return { connected: false, models: [], currentModel: MODEL };
    }
    console.log('[LLM] Health check OK — AI engine connected');
    return { connected: true, models: [MODEL], currentModel: MODEL };
  } catch (err) {
    console.log(`[LLM] Health check error: ${err instanceof Error ? err.message : err}`);
    return { connected: false, models: [], currentModel: MODEL };
  }
}

async function callLlama(prompt: string, systemPrompt: string, maxTokens?: number): Promise<string> {
  await ensureLlamaRunning();

  const cooldown = computeThermalCooldown();
  if (cooldown > 0) await sleep(cooldown);

  const effectiveMax = adaptiveMaxTokens(maxTokens);

  await acquireInferenceLock();
  holdServer();
  const startTime = Date.now();

  try {
    const body: Record<string, unknown> = {
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt },
      ],
      temperature: 0.7,
      top_p: 0.9,
      repeat_penalty: 1.1,
      min_tokens: 40,
      stream: false,
    };
    if (effectiveMax) body.max_tokens = effectiveMax;

    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error(`LLM error: ${res.status}`);
    }

    const data = await res.json();
    const duration = Date.now() - startTime;
    console.log(`[LLM] Inference complete: ${duration}ms`);
    return data.choices[0].message.content;
  } finally {
    recordInferenceTime(Date.now() - startTime);
    releaseInferenceLock();
    releaseServer();
  }
}

async function* callLlamaStream(prompt: string, systemPrompt: string, maxTokens?: number): AsyncGenerator<string> {
  await ensureLlamaRunning();

  const cooldown = computeThermalCooldown();
  if (cooldown > 0) await sleep(cooldown);

  const effectiveMax = adaptiveMaxTokens(maxTokens);

  await acquireInferenceLock();
  holdServer();
  const startTime = Date.now();

  try {
    const body: Record<string, unknown> = {
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt },
      ],
      temperature: 0.7,
      top_p: 0.9,
      repeat_penalty: 1.1,
      min_tokens: 40,
      stream: true,
    };
    if (effectiveMax) body.max_tokens = effectiveMax;

    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error(`LLM error: ${res.status}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let accumulated = '';
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        if (!trimmed.startsWith('data: ')) continue;
        const payload = trimmed.slice(6);
        if (payload === '[DONE]') continue;
        try {
          const data = JSON.parse(payload);
          const content = data.choices?.[0]?.delta?.content;
          if (content) {
            accumulated += content;
            yield accumulated;
          }
        } catch {}
      }
    }
  } finally {
    recordInferenceTime(Date.now() - startTime);
    releaseInferenceLock();
    releaseServer();
  }
}

async function callLLM(prompt: string, systemPrompt: string, maxTokens?: number): Promise<string> {
  return callLlama(prompt, systemPrompt, maxTokens);
}

async function* streamLLM(prompt: string, systemPrompt: string, maxTokens?: number): AsyncGenerator<string> {
  yield* callLlamaStream(prompt, systemPrompt, maxTokens);
}

export function cleanInputText(text: string): string {
  return text
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/[\u200B-\u200D\uFEFF\u00AD]/g, '')
    .replace(/[\u2013\u2014\u2015]/g, '-')
    .replace(/[\u2018\u2019\u201A]/g, "'")
    .replace(/[\u201C\u201D\u201E]/g, '"')
    .replace(/\t/g, ' ')
    .replace(/[ ]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .split('\n')
    .filter((line, i, arr) => i === 0 || line.trim() !== arr[i - 1].trim() || line.trim() === '')
    .join('\n')
    .trim();
}

function cleanSummaryOutput(text: string): string {
  return text
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    .replace(/`{1,3}([^`]+)`{1,3}/g, '$1')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/^(\s*)[•\*]\s+/gm, '$1- ')
    .replace(/[<>{}[\]\\|~^]/g, '')
    .replace(/^(Summary|Here is|Here's|The following|Below is)[:\s]*/im, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function cleanExplainOutput(text: string): string {
  return text
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    .replace(/`{1,3}([^`]+)`{1,3}/g, '$1')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/^[\s]*[•\-\*]\s+/gm, '')
    .replace(/^[\s]*\d+\.\s+(?!["'])/gm, '')
    .replace(/[<>{}[\]\\|~^]/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+/g, ' ')
    .trim();
}

const MAX_OUTPUT_TOKENS = 200;

const SYSTEM_PROMPT = `You are a world-class analyst. Produce comprehensive, insightful summaries that explain the meaning and significance of the content.`;

// --- Extractive-Abstractive Hierarchical Summarization Pipeline
// Centroid-based: TextRank select -> embed -> k-means cluster -> per-cluster LLM -> global synthesis.
// 5-9 LLM calls total.

// --- Sentence Segmentation

function smartSplitSentences(text: string): string[] {
  return text
    .replace(/\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\./g, '$1\u0000')
    .split(/(?<=[.!?])\s+/)
    .map(s => s.replace(/\u0000/g, '.').trim())
    .filter(s => s.length > 0);
}

// --- Summary Validation

function validateSummary(summary: string, _source: string): string {
  if (!summary || summary.trim().length < 10) return '';

  const sentences = summary.split(/(?<=[.!?])\s+/);
  const seen = new Set<string>();
  const deduped: string[] = [];
  for (const s of sentences) {
    const key = s.toLowerCase().trim();
    if (key.length < 5) continue;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(s);
  }

  let result = deduped.join(' ')
    .replace(/â€™/g, "'").replace(/â€œ/g, '"').replace(/â€/g, '"')
    .replace(/Ã©/g, 'é').replace(/ï¿½/g, '').replace(/Â/g, '');

  return cleanSummaryOutput(result);
}

// --- Progressive Summarization (Public API)
// Processes document in ~300-word blocks, yielding structured events for streaming UI.

export type SummaryEvent =
  | { type: 'progress'; message: string }
  | { type: 'blockStream'; text: string; index: number; total: number }  // streaming block text (typing animation)
  | { type: 'blockDone'; summary: string; index: number; total: number } // block complete
  | { type: 'final'; text: string }; // accumulated final summary text

function splitIntoBlocks(text: string, targetWords = 300): string[] {
  const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
  const blocks: string[] = [];
  let current: string[] = [];
  let currentWords = 0;

  for (const para of paragraphs) {
    const paraWords = para.split(/\s+/).length;

    // If single paragraph exceeds target, split by sentences
    if (paraWords > targetWords * 1.5) {
      if (current.length > 0) {
        blocks.push(current.join('\n\n'));
        current = [];
        currentWords = 0;
      }
      const sentences = smartSplitSentences(para);
      let sentBuf: string[] = [];
      let sentWords = 0;
      for (const s of sentences) {
        const sw = s.split(/\s+/).length;
        if (sentWords + sw > targetWords && sentBuf.length > 0) {
          blocks.push(sentBuf.join(' '));
          sentBuf = [];
          sentWords = 0;
        }
        sentBuf.push(s);
        sentWords += sw;
      }
      if (sentBuf.length > 0) blocks.push(sentBuf.join(' '));
      continue;
    }

    if (currentWords + paraWords > targetWords && current.length > 0) {
      blocks.push(current.join('\n\n'));
      current = [];
      currentWords = 0;
    }
    current.push(para);
    currentWords += paraWords;
  }
  if (current.length > 0) blocks.push(current.join('\n\n'));

  return blocks;
}

// Split text into blocks (exported so the route can use it)
export function splitNoteIntoBlocks(text: string): string[] {
  return splitIntoBlocks(cleanInputText(text), 300);
}

// Summarize a single batch of blocks (startIdx to startIdx+batchSize-1)
// Returns block summaries only. No final synthesis.
export async function* summarizeBatchStream(
  blocks: string[],
  startIdx: number,
  batchSize: number,
  previousSummaries: string[],
): AsyncGenerator<SummaryEvent> {
  await ensureLlamaRunning();

  const endIdx = Math.min(startIdx + batchSize, blocks.length);
  const total = blocks.length;
  const currentSummaries = [...previousSummaries];

  for (let i = startIdx; i < endIdx; i++) {
    yield { type: 'progress', message: `Reading section ${i + 1} of ${total}...` };

    let context = '';
    if (currentSummaries.length > 0) {
      const recent = currentSummaries.slice(-2).join('\n');
      context = `Previous context (do not repeat):\n${recent}\n\n`;
    }

    const prompt = `${context}Summarize this section in 3-5 sentences. Preserve ALL names of people, organizations, and relationships. Explain the meaning, not just the facts.\n\nSection ${i + 1} of ${total}:\n${blocks[i]}\n\nSummary:`;

    // Stream each block so the frontend shows typing animation
    let finalText = '';
    for await (const chunk of streamLLM(prompt, SYSTEM_PROMPT, 200)) {
      finalText = cleanSummaryOutput(chunk);
      yield { type: 'blockStream', text: finalText, index: i, total };
    }
    currentSummaries.push(finalText);
    yield { type: 'blockDone', summary: finalText, index: i, total };

    if (i < endIdx - 1) await new Promise(r => setTimeout(r, 200));
  }
}

// Generate the final synthesis from all block summaries
export async function* synthesizeFinalStream(allSummaries: string[]): AsyncGenerator<SummaryEvent> {
  await ensureLlamaRunning();

  yield { type: 'progress', message: 'Writing final summary...' };

  const sections = allSummaries.map((s, i) => `[Section ${i + 1}] ${s}`).join('\n\n');
  const prompt = `You are writing a comprehensive summary of an entire document.\n\nSection summaries:\n${sections}\n\nWrite a detailed, coherent summary (aim for 400-800 words) that:\n- Explains the overall topic and purpose\n- Identifies ALL key people, organizations, and relationships\n- Covers every section — do not skip any\n- Does NOT repeat information\n- Reads as a single coherent narrative\n\nSummary:`;

  for await (const chunk of streamLLM(prompt, SYSTEM_PROMPT, 1300)) {
    yield { type: 'final', text: cleanSummaryOutput(chunk) };
  }
}

// Short note: single direct summarization
export async function* directSummarizeStream(text: string): AsyncGenerator<SummaryEvent> {
  await ensureLlamaRunning();
  yield { type: 'progress', message: 'Summarizing...' };
  const cleaned = cleanInputText(text);
  const prompt = `Explain what this note is about. Identify the main topic, key people, organizations, and relationships.\n\nNote:\n${cleaned}\n\nSummary:`;
  for await (const chunk of streamLLM(prompt, SYSTEM_PROMPT, 500)) {
    yield { type: 'final', text: cleanSummaryOutput(chunk) };
  }
}

// Legacy non-streaming API (used by route.ts POST handler)
export async function summarizeText(text: string): Promise<{ summary: string; keyPoints: string[] } | null> {
  const cleaned = cleanInputText(text);
  const blocks = splitNoteIntoBlocks(cleaned);
  if (blocks.length <= 1) {
    await ensureLlamaRunning();
    const prompt = `Explain what this note is about. Identify the main topic, key people, organizations, and relationships.\n\nNote:\n${cleaned}\n\nSummary:`;
    const response = await callLLM(prompt, SYSTEM_PROMPT, 500);
    return { summary: validateSummary(response, cleaned) || cleaned.slice(0, 500), keyPoints: [] };
  }
  // Process all blocks + synthesize
  const summaries: string[] = [];
  for await (const event of summarizeBatchStream(blocks, 0, blocks.length, [])) {
    if (event.type === 'blockDone') summaries.push(event.summary);
  }
  let final = '';
  for await (const event of synthesizeFinalStream(summaries)) {
    if (event.type === 'final') final = event.text;
  }
  return { summary: validateSummary(final, cleaned) || summaries.join(' '), keyPoints: [] };
}

export async function summarizeChunks(chunks: string[]): Promise<{ summary: string; keyPoints: string[] } | null> {
  return summarizeText(chunks.join('\n\n'));
}

// --- Extractive-Abstractive Pipeline
// Segment -> rank -> cluster -> per-cluster LLM -> entity extraction -> global synthesis

const EXPLAIN_PROMPT = (sentenceBlock: string, count: number) =>
`You are an expert tutor explaining text to a curious, intelligent reader.

For each numbered sentence below, write a thorough explanation that helps the reader truly understand it.

Format your response EXACTLY like this — one block per sentence:
[1] Your explanation for sentence 1 here.
[2] Your explanation for sentence 2 here.
...and so on.

For each explanation:
- Explain what the sentence means in its full context
- Discuss WHY this matters — what are the implications or significance?
- Connect it to broader concepts, real-world applications, or related ideas when relevant
- If the sentence contains technical terms, define them clearly
- Each explanation should be thorough (3-5 sentences) to provide real understanding
- Write in clear, accessible language
- Do NOT repeat or paraphrase the original sentence — explain the meaning behind it

You MUST cover ALL ${count} sentences. Do not skip any.

Sentences:
${sentenceBlock}`;

export async function explainText(text: string): Promise<{ explanation: string } | null> {
  try {
    const cleaned = cleanInputText(text);
    const sentences = cleaned
      .split(/(?<=[.!?])\s+/)
      .filter(s => s.trim().length > 10)
      .slice(0, 15);

    const sentenceBlock = sentences
      .map((s, i) => `[${i + 1}] ${s.trim()}`)
      .join('\n');

    const explanation = await callLLM(EXPLAIN_PROMPT(sentenceBlock, sentences.length), SYSTEM_PROMPT);
    return { explanation: cleanExplainOutput(explanation) };
  } catch (err) {
    console.error('[LLM] Explain error:', err);
    throw err;
  }
}

export async function* explainTextStream(text: string): AsyncGenerator<string> {
  const cleaned = cleanInputText(text);
  const sentences = cleaned
    .split(/(?<=[.!?])\s+/)
    .filter(s => s.trim().length > 10)
    .slice(0, 15);

  const sentenceBlock = sentences
    .map((s, i) => `[${i + 1}] ${s.trim()}`)
    .join('\n');

  for await (const chunk of streamLLM(EXPLAIN_PROMPT(sentenceBlock, sentences.length), SYSTEM_PROMPT)) {
    yield cleanExplainOutput(chunk);
  }
}

export function chunkByWords(text: string, wordsPerChunk = 150): string[] {
  const cleaned = cleanInputText(text);
  const words = cleaned.split(/\s+/).filter(w => w.length > 0);
  const sections: string[] = [];

  for (let i = 0; i < words.length; i += wordsPerChunk) {
    const chunk = words.slice(i, i + wordsPerChunk).join(' ').trim();
    if (chunk.length > 5) sections.push(chunk);
  }

  return sections;
}

export function chunkForExplain(text: string): string[] {
  return chunkByWords(text, 100);
}

const EXPLAIN_SECTION_PROMPT = (sectionText: string, sectionIndex: number, totalSections: number, previousContext: string, surroundingContext?: string) => {
  let prompt = `You are a concise tutor. This is section ${sectionIndex + 1} of ${totalSections}.\n\n`;

  if (previousContext) {
    prompt += `PRIOR EXPLANATION CONTEXT (do not repeat):\n${previousContext}\n\n`;
  }

  if (surroundingContext) {
    prompt += `SURROUNDING DOCUMENT CONTEXT (use for understanding, do not explain this):\n${surroundingContext}\n\n`;
  }

  prompt += `Write a BRIEF explanation of the section below. STRICT RULES:
- Your ENTIRE response must be under 150 words
- Write ONE short paragraph covering the key meaning
- Define technical terms inline if any
- Do NOT list or number sentences individually
- Do NOT repeat or paraphrase the original text
- Be direct and concise — no filler

Section:
"""
${sectionText}
"""`;

  return prompt;
};

export async function* explainSectionStream(
  sectionText: string,
  sectionIndex: number,
  totalSections: number,
  previousContext: string,
  surroundingContext?: string,
): AsyncGenerator<string> {
  const prompt = EXPLAIN_SECTION_PROMPT(sectionText, sectionIndex, totalSections, previousContext, surroundingContext);
  for await (const chunk of streamLLM(prompt, SYSTEM_PROMPT, MAX_OUTPUT_TOKENS)) {
    yield cleanExplainOutput(chunk);
  }
}

export type QueryType = 'greeting' | 'casual' | 'summarize' | 'explain' | 'document-question';

const GREETING_PATTERNS = /^(hi|hello|hey|good\s*(morning|afternoon|evening|day)|howdy|what'?s\s*up|yo|sup|hola|greetings)\b/i;
const CASUAL_PATTERNS = /^(how\s+are\s+you|what\s+do\s+you\s+do|who\s+are\s+you|tell\s+me\s+(a\s+joke|about\s+yourself)|what'?s?\s+your\s+name|thanks?|thank\s+you|bye|goodbye|see\s+you|ok|okay|cool|nice|great)\b/i;
const SUMMARIZE_PATTERNS = /\b(summarize|summary|summarise|give\s+me\s+a\s+summary|overview|brief|tl;?dr|key\s*points|main\s*points|recap)\b/i;
const EXPLAIN_PATTERNS = /\b(explain|explanation|what\s+does\s+(this|it|that)\s+mean|break\s+(it\s+)?down|elaborate|clarify|simplify|make\s+it\s+simple)\b/i;

export function classifyQuery(message: string): QueryType {
  const trimmed = message.trim();
  if (GREETING_PATTERNS.test(trimmed)) return 'greeting';
  if (CASUAL_PATTERNS.test(trimmed)) return 'casual';
  if (SUMMARIZE_PATTERNS.test(trimmed)) return 'summarize';
  if (EXPLAIN_PATTERNS.test(trimmed)) return 'explain';
  return 'document-question';
}

const GREETING_RESPONSES = [
  "Hello! I'm your document assistant. Ask me anything about the current document.",
  "Hi there! I can help you understand, summarize, or explain the content of this document. What would you like to know?",
  "Hey! I'm here to help with document analysis. Ask me a question about the content.",
];

export function getGreetingResponse(): string {
  return GREETING_RESPONSES[Math.floor(Math.random() * GREETING_RESPONSES.length)];
}

// --- Mistral 7B Chat Pipeline (8-step)
// frustration -> correction -> intent -> rewrite -> retrieve -> assemble -> generate -> validate

type ChatMessage = { role: string; content: string };
type ChatIntent = 'EXHAUSTIVE_LIST' | 'PERSON_QUERY' | 'FACT_LOOKUP' | 'YES_NO' | 'SUMMARY_REQUEST' | 'COMPARISON' | 'FOLLOW_UP' | 'CORRECTION' | 'TOPIC_CHANGE';

const TOKEN_BUDGET = { system: 250, summary: 100, context: 1400, recentChat: 400, query: 150 };
const CHAT_MAX_TOKENS = 1024;
const MAX_RECENT = 6; // 3 turns
const KV_RESET_INTERVAL = 6;

// Session state
let cachedSummary = '';
let cachedSummaryLen = 0;
let chatTurns = 0;
const excludedEntities: Set<string> = new Set();
const recentResponses: string[] = [];

// Two-tier summary: Tier 1 = rolling context (compressed aggressively).
// Tier 2 = pinned facts (persist until explicitly changed by user).
const pinnedFacts: string[] = [];

// --- Entity Context Builder
// Builds structured relationship graph from entity index to prevent entity relation collapse.

// --- Helpers

function truncTk(text: string, max: number): string {
  const w = text.split(/\s+/), m = Math.floor(max / 1.3);
  return w.length <= m ? text : w.slice(0, m).join(' ') + '...';
}

async function quickCall(prompt: string, maxTk = 50): Promise<string> {
  try {
    const r = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: prompt }], temperature: 0.1, max_tokens: maxTk, repeat_penalty: 1.1, stream: false }),
    });
    if (r.ok) { const d = await r.json(); return (d.choices?.[0]?.message?.content || '').trim(); }
  } catch {}
  return '';
}

// --- Distress Detection (pre-pipeline, no model call)
// Legal/ethical requirement per ISACA 2025.

const DISTRESS_RE = /\b(hate my life|want to die|kill myself|end it all|no point in living|suicide|self.?harm|hurt myself|nobody cares|worthless|hopeless|can'?t go on|don'?t want to live)\b/i;

const DISTRESS_RESPONSE = "It sounds like you're going through something really difficult. I'm a notes assistant and not the right resource for this, but please talk to someone who can help.\n\n988 Suicide and Crisis Lifeline: call or text 988 (US)\nCrisis Text Line: text HOME to 741741\n\nYou're not alone, and these feelings can get better with support.";

function detectDistress(msg: string): boolean {
  return DISTRESS_RE.test(msg);
}

// --- Frustration Detection

const FRUSTRATION_RE = /\b(dumb|stupid|wtf|idiot|useless|wrong again|i already said|i told you|i just asked|for the .* time|forget it|never mind|are you even reading|not what i asked|that's not it|no that's wrong|you're wrong|still wrong)\b/i;

function detectFrustration(msg: string): boolean {
  return FRUSTRATION_RE.test(msg);
}

// --- Correction Detection

function detectCorrection(msg: string, lastResponse: string): { isCorrection: boolean; assertion: string } {
  if (!lastResponse) return { isCorrection: false, assertion: '' };

  // Regex-based correction detection — eliminates a 50-token model call
  // that adds 1-2s latency and can produce cascade errors.
  const CORRECTION_RE = /\b(no[,.]?\s|wrong|incorrect|not right|that's not|you're wrong|actually|the correct|it's actually|it should be|that is wrong)\b/i;
  if (CORRECTION_RE.test(msg)) {
    const corrected = msg.replace(/\b(no|wrong|incorrect|not|that's not|you're wrong|actually)\b/gi, '').trim();
    return { isCorrection: true, assertion: corrected || msg };
  }
  return { isCorrection: false, assertion: '' };
}

// --- Intent Classification

function classifyIntent(msg: string, isCorrection: boolean): ChatIntent {
  if (isCorrection) return 'CORRECTION';
  const m = msg.toLowerCase();
  if (/\b(all|every|list|each|complete list|how many)\b/.test(m) && /\b(mention|name|college|university|tool|person|item|instance|reference)\b/.test(m)) return 'EXHAUSTIVE_LIST';
  if (/\bwho is\b|\bwho was\b|\babout .+ person\b/.test(m)) return 'PERSON_QUERY';
  if (/\bcompare\b|\bversus\b|\bvs\b|\bdifference between\b/.test(m)) return 'COMPARISON';
  if (/\bsummar|\boverview\b|\bbrief\b|\btl;?dr\b/.test(m)) return 'SUMMARY_REQUEST';
  // Yes/No: starts with "is", "does", "did", "was", "are", "has", "can", "will"
  if (/^(is|does|did|was|are|has|have|can|could|will|would|should)\b/i.test(m.trim())) return 'YES_NO';
  if (/\b(this|that|it|the other|here|there|above|those)\b/.test(m) && m.split(/\s+/).length < 10) return 'FOLLOW_UP';
  if (/\b(forget|stop talking about|different topic|change subject|anyway|moving on)\b/.test(m)) return 'TOPIC_CHANGE';
  return 'FACT_LOOKUP';
}

// --- Query Rewriting

async function rewriteQuery(raw: string, recent: ChatMessage[]): Promise<string> {
  if (recent.length === 0 || raw.split(/\s+/).length > 20) return raw;
  const ctx = recent.slice(-6).map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content.slice(0, 100)}`).join('\n');
  const result = await quickCall(
    `Rewrite this message as a clear standalone question. Replace all pronouns and references ("this," "that," "it," "the other one") with the actual thing being referenced.\n\nRecent conversation:\n${ctx}\n\nUser message: ${raw}\n\nOutput only the rewritten question:`,
    60
  );
  if (result && result.length > 5 && result.length < 300) {
    console.log(`[Chat] Rewrite: "${raw}" → "${result}"`);
    return result;
  }
  return raw;
}

// --- Context Compression

// --- Summary Management

async function manageSummary(history: ChatMessage[], intent: ChatIntent, frustrated: boolean): Promise<{ summary: string; recent: ChatMessage[] }> {
  // Wipe rolling summary on topic change or frustration (pinned facts survive)
  if (intent === 'TOPIC_CHANGE' || frustrated) {
    cachedSummary = '';
    cachedSummaryLen = 0;
  }

  // Extract pinned facts from corrections
  if (intent === 'CORRECTION' && history.length > 0) {
    const lastUser = history.filter(m => m.role === 'user').pop();
    if (lastUser) {
      const fact = lastUser.content.replace(/\b(no|wrong|incorrect|actually|the correct|it's)\b/gi, '').trim();
      if (fact.length > 10 && fact.length < 200) {
        // Avoid duplicate pins
        if (!pinnedFacts.some(p => p.toLowerCase() === fact.toLowerCase())) {
          pinnedFacts.push(fact);
          if (pinnedFacts.length > 5) pinnedFacts.shift(); // max 5 pinned facts
          console.log(`[Chat] Pinned fact: "${fact}"`);
        }
      }
    }
  }

  if (history.length <= MAX_RECENT) {
    const s = buildTwoTierSummary(frustrated ? '' : cachedSummary);
    return { summary: s, recent: history };
  }

  const older = history.slice(0, -MAX_RECENT);
  const recent = history.slice(-MAX_RECENT);

  if (older.length > cachedSummaryLen || intent === 'CORRECTION' || intent === 'TOPIC_CHANGE') {
    const convText = older.map(m => `${m.role === 'user' ? 'User' : 'Asst'}: ${m.content.slice(0, 100)}`).join('\n');
    const raw = await quickCall(
      `Summarize this conversation in 2-3 sentences. Include: what the user wanted, answers given, any corrections. Do not include entity names unless confirmed correct.\n\n${convText}\n\nSummary:`,
      80
    );
    cachedSummary = truncTk(raw || '', TOKEN_BUDGET.summary - (pinnedFacts.length * 15));
    cachedSummaryLen = older.length;

    for (const entity of excludedEntities) {
      cachedSummary = cachedSummary.replace(new RegExp(entity, 'gi'), '[removed]');
    }
  }

  const truncated = recent.map(m => ({ ...m, content: truncTk(m.content, TOKEN_BUDGET.recentChat / MAX_RECENT) }));
  const s = buildTwoTierSummary(frustrated ? '' : cachedSummary);
  return { summary: s, recent: truncated };
}

function buildTwoTierSummary(rollingSummary: string): string {
  const parts: string[] = [];
  if (rollingSummary) parts.push(rollingSummary);
  if (pinnedFacts.length > 0) {
    parts.push('Key facts: ' + pinnedFacts.join('. '));
  }
  return truncTk(parts.join('\n'), TOKEN_BUDGET.summary);
}

// --- Prompt Assembly & Response Design

// --- Chat System Prompt (Q&A focused)
const CHAT_SYSTEM = `You are Epito, an AI assistant that answers questions about the user's notes. Rules:
1. Answer the question directly using information from the provided notes
2. When mentioning people or organizations, explain their relationship (e.g., "X is affiliated with Y")
3. If the answer is not in the notes, say "I don't have that information in your notes"
4. Never copy text verbatim — explain in your own words
5. Be specific — cite names, dates, and facts from the notes`;

const CHAT_GROUNDING = `Answer using only the notes provided. Be specific and explain relationships between entities.`;

// Intent-specific prompt additions (one-liner per intent, appended to user query)
const INTENT_PROMPTS: Record<ChatIntent, string> = {
  FACT_LOOKUP: 'Answer with the specific fact and where in the notes it appears.',
  YES_NO: 'Answer YES or NO first, then give evidence from the notes.',
  EXHAUSTIVE_LIST: 'List every instance found. For each, state its relationship to other entities (person→institution, tool→project). Give the total count.',
  PERSON_QUERY: 'Describe this person: their role, affiliations, and relationships to other entities in the notes.',
  COMPARISON: 'Compare using specific facts from the notes. State differences clearly.',
  SUMMARY_REQUEST: 'Give a brief overview of the main topic and key entities in the notes.',
  FOLLOW_UP: '',
  CORRECTION: 'The user corrected your previous answer. Provide the correct information.',
  TOPIC_CHANGE: '',
};

const INTENT_MAX_TOKENS: Record<ChatIntent, number> = {
  FACT_LOOKUP: 1024,
  YES_NO: 1024,
  EXHAUSTIVE_LIST: 1024,
  PERSON_QUERY: 1024,
  COMPARISON: 1024,
  SUMMARY_REQUEST: 1024,
  FOLLOW_UP: 1024,
  CORRECTION: 1024,
  TOPIC_CHANGE: 1024,
};

function buildMessages(
  context: string, summary: string, recent: ChatMessage[], query: string,
  flags: { frustrated: boolean; isCorrection: boolean; intent: ChatIntent },
): Array<{ role: string; content: string }> {
  let sys = CHAT_SYSTEM;

  if (flags.frustrated) {
    sys = `Give a direct answer. Do not apologize. Just answer correctly.\n\n${sys}`;
  }
  for (const e of excludedEntities) {
    sys += `\nDo not mention ${e} unless the user specifically asks about them.`;
  }

  sys = truncTk(sys, TOKEN_BUDGET.system);

  if (summary) {
    sys += `\n\nPrevious conversation context:\n${summary}`;
  }

  if (context && context.trim()) {
    sys += `\n\n### Notes\n${context}`;
  } else {
    sys += `\n\n### Notes\nNo matching notes found for this question.`;
  }

  const msgs: Array<{ role: string; content: string }> = [{ role: 'system', content: sys }];
  for (const m of recent) msgs.push({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content });

  const intentPrompt = INTENT_PROMPTS[flags.intent] || '';
  const queryWithGrounding = intentPrompt
    ? `${query}\n\n${intentPrompt}\n\n${CHAT_GROUNDING}`
    : `${query}\n\n${CHAT_GROUNDING}`;
  msgs.push({ role: 'user', content: queryWithGrounding });

  return msgs;
}

// --- Output Validation

// --- Grounding Check (post-generation entity validation)

function groundingCheck(response: string, context: string): string[] {
  if (!context || !response) return [];
  const contextLower = context.toLowerCase();
  const ungrounded: string[] = [];

  // Extract named entities (capitalized multi-word phrases)
  const entityRe = /[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+/g;
  let m;
  while ((m = entityRe.exec(response)) !== null) {
    const entity = m[0];
    if (!contextLower.includes(entity.toLowerCase())) {
      // Fuzzy: check if any 3+ word subsequence matches
      const words = entity.toLowerCase().split(/\s+/);
      const found = words.some(w => w.length > 3 && contextLower.includes(w));
      if (!found) ungrounded.push(entity);
    }
  }

  // Extract numbers not in context
  const numRe = /\b\d[\d,.]+\b/g;
  while ((m = numRe.exec(response)) !== null) {
    if (!context.includes(m[0])) ungrounded.push(m[0]);
  }

  return ungrounded;
}

// --- Core Pipeline

function getParams(frustrated: boolean, isCorrection: boolean, intent: ChatIntent) {
  const maxTk = INTENT_MAX_TOKENS[intent] || CHAT_MAX_TOKENS;
  if (frustrated) return { temperature: 0.5, repeat_penalty: 1.15, presence_penalty: 0.1, top_p: 0.9, top_k: 50, max_tokens: maxTk };
  if (isCorrection) return { temperature: 0.6, repeat_penalty: 1.15, presence_penalty: 0.1, top_p: 0.9, top_k: 50, max_tokens: maxTk };
  return { temperature: 0.7, repeat_penalty: 1.1, presence_penalty: 0.0, top_p: 0.9, top_k: 50, max_tokens: maxTk };
}

// --- MSR-RAG Chat Pipeline (Multi-Stage Reasoning Retrieval)
// Query Understanding -> Retrieval -> Reasoning -> Targeted Retrieval -> Assembly -> Synthesis -> Verification

// --- Query Understanding (zero LLM)
// Extracts intent, entities, relationship patterns, question type, and implicit entity types.

type QuestionType = 'entity_lookup' | 'explanation' | 'comparison' | 'list' | 'yes_no' | 'relationship' | 'general';

function understandQuery(query: string): {
  intent: ChatIntent;
  queryEntities: string[];
  implicitEntityTypes: string[];
  questionType: QuestionType;
  relationshipQuery: boolean;
} {
  const q = query.toLowerCase();
  const intent = classifyIntent(query, false);

  // Extract explicit entity mentions (capitalized multi-word phrases)
  const entityRe = /[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|and|for|Sir|General))*(?:\s+(?:University|Institute|College|Academy|Inc|Corp|Ltd))?/g;
  const queryEntities: string[] = [];
  let m;
  while ((m = entityRe.exec(query)) !== null) {
    if (m[0].length > 3) queryEntities.push(m[0]);
  }

  // Detect implicit entity types from query keywords
  const implicitEntityTypes: string[] = [];
  if (/\b(universit|college|institut|school|academ)/i.test(q)) implicitEntityTypes.push('ORG');
  if (/\b(person|people|author|researcher|who|professor|student|scientist)/i.test(q)) implicitEntityTypes.push('PERSON');
  if (/\b(compan|organization|firm|corp|business)/i.test(q)) implicitEntityTypes.push('ORG');
  if (/\b(tool|library|framework|software|technolog)/i.test(q)) implicitEntityTypes.push('TOOL');

  // Detect relationship queries ("who is associated with", "affiliated with", etc.)
  const relationshipQuery = /\b(associated|affiliated|related|connected|linked|belong|from|at|work)/i.test(q);

  // Classify question type
  let questionType: QuestionType = 'general';
  if (/\b(list|all|every|how many|what are|which|names?|mentioned)/i.test(q)) questionType = 'list';
  else if (/\b(compare|versus|vs|difference|between)/i.test(q)) questionType = 'comparison';
  else if (/\b(explain|what does|what is|meaning|why|describe|about)/i.test(q)) questionType = 'explanation';
  else if (/\b(who is|who was|who are|about .+ person)/i.test(q)) questionType = 'entity_lookup';
  else if (/^(is|does|did|was|are|has|can|will)\b/i.test(q.trim())) questionType = 'yes_no';
  else if (relationshipQuery) questionType = 'relationship';

  return { intent, queryEntities, implicitEntityTypes, questionType, relationshipQuery };
}

// --- Reasoning (gap analysis on stage 1 context)
// Identifies missing entities, thin context, relationship gaps; produces follow-up queries.

function reasonAboutContext(
  query: string,
  stage1Context: string[],
  queryEntities: string[],
  implicitEntityTypes: string[],
  questionType: QuestionType,
): { missingEntities: string[]; followUpQueries: string[]; entityTypeQueries: string[]; hasSufficientContext: boolean } {
  if (stage1Context.length === 0) {
    return {
      missingEntities: queryEntities,
      followUpQueries: [query],
      entityTypeQueries: implicitEntityTypes,
      hasSufficientContext: false,
    };
  }

  const combined = stage1Context.join(' ');
  const combinedLower = combined.toLowerCase();
  const wordCount = combinedLower.split(/\s+/).length;

  // Check which explicit entities are missing
  const missingEntities = queryEntities.filter(e => !combinedLower.includes(e.toLowerCase()));

  // For list/relationship queries, check if context has enough entity density
  const entityDensity = (combined.match(/[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+/g) || []).length;
  const entityDensityLow = (questionType === 'list' || questionType === 'relationship') && entityDensity < 3;

  // Context sufficiency: need enough words AND all entities AND decent entity density
  const hasSufficientContext = wordCount > 80 && missingEntities.length === 0 && !entityDensityLow;

  // Build follow-up queries
  const followUpQueries: string[] = [];

  // Query for each missing explicit entity
  for (const entity of missingEntities) followUpQueries.push(entity);

  // For list queries with thin entity density, broaden the search
  if (entityDensityLow) {
    const keywords = query.split(/\s+/).filter(w => w.length > 4 && !/^(what|where|when|which|about|their|these|those|every|mentioned)$/i.test(w.toLowerCase()));
    if (keywords.length > 0) followUpQueries.push(keywords.slice(0, 3).join(' '));
  }

  // If context is very thin overall, add the raw query as fallback
  if (wordCount < 40 && followUpQueries.length === 0) {
    followUpQueries.push(query);
  }

  // Entity type queries — search the entity DB for these types
  const entityTypeQueries = entityDensityLow ? implicitEntityTypes : [];

  if (!hasSufficientContext) {
    console.log(`[MSR-RAG] Reasoning: missing=${missingEntities.length} entities, density=${entityDensity}, words=${wordCount}, followUps=${followUpQueries.length}, typeQueries=${entityTypeQueries.length}`);
  }

  return { missingEntities, followUpQueries, entityTypeQueries, hasSufficientContext };
}

// --- Stage 2 Targeted Retrieval
// Three strategies: contextual search, entity DB lookup, entity type search.

async function stage2Retrieval(
  sourceId: string | null,
  followUpQueries: string[],
  missingEntities: string[],
  entityTypeQueries: string[],
  existingContexts: string[],
): Promise<string[]> {
  const newContexts: string[] = [];
  const seen = new Set(existingContexts.map(c => c.slice(0, 100).toLowerCase()));

  // Strategy 1: Contextual retrieval for follow-up queries
  for (const fq of followUpQueries.slice(0, 3)) {
    try {
      const results = await contextualRetrieveForChat(sourceId, fq, 4);
      for (const ctx of results.contexts) {
        const key = ctx.slice(0, 100).toLowerCase();
        if (!seen.has(key)) {
          seen.add(key);
          newContexts.push(ctx);
        }
      }
    } catch {}
  }

  // Strategy 2: Entity DB search for specific missing entities
  for (const entity of missingEntities.slice(0, 5)) {
    try {
      const dbEntities = db.searchEntities(entity);
      for (const e of dbEntities) {
        if (e.evidence) {
          const key = e.evidence.slice(0, 100).toLowerCase();
          if (!seen.has(key)) {
            seen.add(key);
            newContexts.push(`${e.entity_name} (${e.entity_type}): ${e.evidence}`);
          }
        }
      }
    } catch {}
  }

  // Strategy 3: Entity type search (e.g. find all ORGs when query asks about "universities")
  for (const entityType of entityTypeQueries.slice(0, 2)) {
    try {
      const dbEntities = db.searchEntities('%', entityType);
      for (const e of dbEntities.slice(0, 10)) {
        if (e.evidence) {
          const key = e.entity_name.toLowerCase();
          if (!seen.has(key)) {
            seen.add(key);
            newContexts.push(`${e.entity_name} (${e.entity_type}): ${e.evidence}`);
          }
        }
      }
    } catch {}
  }

  console.log(`[MSR-RAG] Stage 2: ${newContexts.length} additional contexts (queries=${followUpQueries.length}, entities=${missingEntities.length}, types=${entityTypeQueries.length})`);
  return newContexts;
}

// --- Context Assembly
// Entity-aware ranking, sentence dedup, token budget enforcement.

function assembleContext(
  stage1: string[],
  stage2: string[],
  queryEntities: string[],
  _questionType: QuestionType,
  maxTokens: number,
): string {
  const all = [...stage1, ...stage2];
  if (all.length === 0) return '';

  // Score each chunk on multiple signals
  const scored = all.map((chunk, i) => {
    const lower = chunk.toLowerCase();

    // Signal 1: Query entity overlap (strongest signal)
    let entityOverlap = 0;
    for (const e of queryEntities) {
      if (lower.includes(e.toLowerCase())) entityOverlap += 3;
    }

    // Signal 2: Named entity density (entity-rich chunks are more valuable)
    const namedEntities = chunk.match(/[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|and|for))+/g) || [];
    const entityDensity = Math.min(namedEntities.length, 8) * 0.5;

    // Signal 3: Relationship indicators (contains words linking entities)
    const hasRelationship = /\b(affiliated|associated|at|from|founded|professor|student|member|director|researcher|university|institute)\b/i.test(chunk) ? 1.5 : 0;

    // Signal 4: Source stage (stage 1 = broader, stage 2 = targeted — both valuable)
    const sourceBonus = i < stage1.length ? 1 : 1.5; // stage 2 is more targeted

    // Signal 5: Fact density (numbers, dates, specific claims)
    const factDensity = (chunk.match(/\b\d{4}\b|\b\d+%|\b\d+\.\d+/g) || []).length * 0.3;

    return { chunk, score: entityOverlap + entityDensity + hasRelationship + sourceBonus + factDensity };
  });

  scored.sort((a, b) => b.score - a.score);

  // Sentence-level deduplication across all chunks
  const globalSentences = new Set<string>();
  const assembled: string[] = [];
  let totalTokens = 0;

  for (const { chunk } of scored) {
    const sentences = chunk.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 10);
    const unique: string[] = [];
    for (const s of sentences) {
      const key = s.trim().toLowerCase().slice(0, 80);
      if (!globalSentences.has(key)) {
        globalSentences.add(key);
        unique.push(s);
      }
    }
    if (unique.length === 0) continue;
    const text = unique.join(' ');
    const tk = Math.ceil(text.split(/\s+/).length * 1.3);
    if (totalTokens + tk > maxTokens) {
      // Try to fit partial chunk
      const remaining = maxTokens - totalTokens;
      if (remaining > 40) {
        assembled.push(truncTk(text, remaining));
        totalTokens = maxTokens;
      }
      break;
    }
    assembled.push(text);
    totalTokens += tk;
  }

  return assembled.join('\n\n');
}

// Step 6+7+8: Full MSR-RAG streaming chat
export async function* msrChatStream(
  sourceId: string | null,
  userMessage: string,
  history: ChatMessage[],
): AsyncGenerator<string> {
  await ensureLlamaRunning();
  await acquireInferenceLock();
  holdServer();
  const start = Date.now();

  try {
    chatTurns++;

    // Step 0: Distress
    if (detectDistress(userMessage)) { yield DISTRESS_RESPONSE; return; }

    // Step 1: Query Understanding
    const { intent, queryEntities, implicitEntityTypes, questionType, relationshipQuery } = understandQuery(userMessage);
    const frustrated = detectFrustration(userMessage);
    const lastResponse = history.length > 0 ? history[history.length - 1]?.content || '' : '';
    const { isCorrection, assertion } = detectCorrection(userMessage, lastResponse);

    console.log(`[MSR-RAG] Query: "${userMessage}" | intent=${intent} | type=${questionType} | entities=[${queryEntities.join(', ')}] | implicitTypes=[${implicitEntityTypes.join(', ')}] | relationship=${relationshipQuery}`);

    // KV cache management
    if (chatTurns % KV_RESET_INTERVAL === 0) {
      try { await fetch(`${LLAMA_URL}/slots/0?action=erase`, { method: 'POST', signal: AbortSignal.timeout(2000) }); } catch {}
      cachedSummary = ''; cachedSummaryLen = 0;
    }

    // Query rewriting for coreference resolution
    const { summary, recent } = await manageSummary(history, intent, frustrated);
    let rewritten = await rewriteQuery(userMessage, recent);
    if (isCorrection && assertion) rewritten += ` (Correction: ${assertion})`;

    // Step 2: Stage 1 Broad Retrieval (hybrid + RRF + reranking + MMR)
    const stage1Result = await contextualRetrieveForChat(sourceId, rewritten, 8);
    console.log(`[MSR-RAG] Stage 1: ${stage1Result.contexts.length} contexts retrieved`);

    // Step 2.5: Full-Coverage Entity Extraction
    // For entity/list/relationship queries, scan ALL chunks from the source
    // document to extract every entity. Don't rely on retrieval coverage.
    let extractedEntities: { name: string; type: string; evidence: string }[] = [];
    const needsFullCoverage = questionType === 'list' || questionType === 'entity_lookup' || questionType === 'relationship' || implicitEntityTypes.length > 0;

    if (needsFullCoverage && sourceId) {
      // Strategy 1: DB entity index (pre-built during ingestion, covers entire document)
      try {
        const dbEntities = db.getEntitiesByDocument(sourceId);
        if (dbEntities.length > 0) {
          // Filter by implicit type if specified
          let filtered = dbEntities;
          if (implicitEntityTypes.length > 0) {
            filtered = dbEntities.filter(e => implicitEntityTypes.includes(e.entity_type));
            if (filtered.length === 0) filtered = dbEntities;
          }
          // Deduplicate by normalized name
          const seen = new Map<string, typeof filtered[0]>();
          for (const e of filtered) {
            const key = e.entity_name.toLowerCase().trim();
            if (!seen.has(key)) seen.set(key, e);
          }
          extractedEntities = [...seen.values()].map(e => ({ name: e.entity_name, type: e.entity_type, evidence: e.evidence }));
          console.log(`[MSR-RAG] Entity extraction (DB): ${extractedEntities.length} entities from document index`);
        }
      } catch {}

      // Strategy 2: If DB index is empty, scan all chunks directly
      if (extractedEntities.length === 0) {
        try {
          const allChunks = db.getChunksByNote(sourceId);
          const allText = allChunks.map(c => c.content).join('\n');
          const entityPatterns: { type: string; re: RegExp }[] = [
            { type: 'ORG', re: /(?:General\s+(?:Sir\s+)?)?[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|and|for|in|Sir|General))*\s+(?:University|Institute|College|Academy|School)/g },
            { type: 'ORG', re: /University\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g },
            { type: 'ORG', re: /[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Co|Group|Foundation|Technologies|Labs|Systems)\.?)/g },
            { type: 'PERSON', re: /[A-Z][a-z]{1,15}\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{1,15}/g },
          ];
          const seen = new Map<string, { name: string; type: string; evidence: string }>();
          for (const { type, re } of entityPatterns) {
            if (implicitEntityTypes.length > 0 && !implicitEntityTypes.includes(type)) continue;
            let match;
            while ((match = re.exec(allText)) !== null) {
              const name = match[0].trim();
              if (name.length < 4) continue;
              const key = name.toLowerCase();
              if (!seen.has(key)) {
                const start = Math.max(0, allText.lastIndexOf('.', match.index) + 1);
                const end = allText.indexOf('.', match.index + name.length);
                const evidence = allText.slice(start, end > 0 ? end + 1 : match.index + name.length + 80).trim().slice(0, 150);
                seen.set(key, { name, type, evidence });
              }
            }
          }
          extractedEntities = [...seen.values()];
          console.log(`[MSR-RAG] Entity extraction (regex): ${extractedEntities.length} entities from full text scan`);
        } catch {}
      }
    }

    // Step 3: Reasoning — analyze gaps in retrieved context
    const reasoning = reasonAboutContext(rewritten, stage1Result.contexts, queryEntities, implicitEntityTypes, questionType);

    // Step 4: Stage 2 Targeted Retrieval (only if gaps detected)
    let stage2Contexts: string[] = [];
    if (!reasoning.hasSufficientContext) {
      stage2Contexts = await stage2Retrieval(
        sourceId,
        reasoning.followUpQueries,
        reasoning.missingEntities,
        reasoning.entityTypeQueries,
        stage1Result.contexts,
      );
    }

    // Step 5: Build final context within a strict total token budget.
    // Total available: 4096 ctx - 250 system - 100 summary - 400 chat - 150 query - 1024 generation = ~2172
    // Reserve ~800 for entities/headers, ~1200 for retrieved text.
    const CONTEXT_BUDGET = 1200;
    const ENTITY_BUDGET = 600;

    const assembledContext = assembleContext(
      stage1Result.contexts,
      stage2Contexts,
      queryEntities,
      questionType,
      CONTEXT_BUDGET,
    );

    // Entity coverage section (kept within budget)
    let entitySection = '';
    if (extractedEntities.length > 0) {
      const grouped = new Map<string, string[]>();
      for (const e of extractedEntities) {
        if (!grouped.has(e.type)) grouped.set(e.type, []);
        grouped.get(e.type)!.push(e.name);
      }
      const parts: string[] = [];
      for (const [type, names] of grouped) {
        const label = type === 'ORG' ? 'Organizations/Institutions' : type === 'PERSON' ? 'People' : type === 'TOOL' ? 'Tools' : type;
        parts.push(`${label}: ${names.join(', ')}`);
      }
      entitySection = `Entities (${extractedEntities.length}): ${parts.join('. ')}`;

      // Add a few evidence lines, keep it short
      const evidenceLines = extractedEntities
        .filter(e => e.evidence && e.evidence.length > 20)
        .slice(0, 5)
        .map(e => `${e.name}: ${e.evidence.slice(0, 100)}`);
      if (evidenceLines.length > 0) {
        entitySection += '\n' + evidenceLines.join('\n');
      }
      entitySection = truncTk(entitySection, ENTITY_BUDGET);
    }

    // Combine: entity section + note content
    let finalContext = '';
    if (entitySection) finalContext += entitySection + '\n\n';
    finalContext += assembledContext;

    const contextWords = finalContext.split(/\s+/).length;
    console.log(`[MSR-RAG] Context: ${contextWords} words, ${extractedEntities.length} entities (stage1=${stage1Result.contexts.length}, stage2=${stage2Contexts.length})`);

    // Step 6: Prompt Assembly
    const flags = { frustrated, isCorrection, intent };
    const messages = buildMessages(finalContext, summary, recent, rewritten, flags);

    // Step 7: Answer Synthesis — stream generation
    const params = getParams(frustrated, isCorrection, intent);
    const body = { messages, ...params, stream: true };

    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`LLM error: ${res.status}`);

    // Simple single-pass stream. No continuation loop. No minimum.
    const reader = res.body?.getReader();
    if (!reader) throw new Error('No body');
    const decoder = new TextDecoder();
    let accumulated = '', buffer = '';
    let tokenCount = 0;

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
            tokenCount++;
            yield accumulated;
          }
        } catch {}
      }
    }

    // Clean trailing model artifacts.
    // Mistral 7B often outputs "terminated", "</s>", or other meta-text
    // at the end when it hits max_tokens (llama-server b4722 has bug #8856
    // where finish_reason is always "stop" even on token limit cutoff).
    accumulated = accumulated
      .replace(/<\/s>/g, '')
      .replace(/<\|im_end\|>/g, '')
      .replace(/<\|endoftext\|>/g, '')
      .replace(/<\|end\|>/g, '')
      .replace(/\s*\bterminated\b\.?\s*$/i, '')
      .replace(/\s*\bgeneration\s+terminated\b\.?\s*$/i, '')
      .replace(/\s*\bresponse\s+terminated\b\.?\s*$/i, '')
      .replace(/\s*\boutput\s+terminated\b\.?\s*$/i, '')
      .replace(/\s*\.{3,}\s*$/g, '')
      .trim();

    // Detect token limit hit by counting tokens.
    // llama-server b4722 bug #8856: finish_reason is always "stop", never "length".
    // So we check if token count is close to max_tokens (within 90%).
    const maxTk = params.max_tokens || 1024;
    const hitTokenLimit = tokenCount >= maxTk * 0.9;

    if (hitTokenLimit && accumulated.length > 0) {
      accumulated += '\n\n---\n*Word limit reached. Please ask a more specific question for a detailed answer.*';
    }

    // Yield final cleaned version
    yield accumulated;

    // Step 8: Answer Verification (post-generation)
    const ungrounded = groundingCheck(accumulated, assembledContext);
    if (ungrounded.length > 0) {
      console.log(`[MSR-RAG] Verification: ${ungrounded.length} ungrounded claims: ${ungrounded.slice(0, 3).join(', ')}`);
    }

    recentResponses.push(accumulated);
    if (recentResponses.length > 5) recentResponses.shift();

    console.log(`[MSR-RAG] ${Date.now() - start}ms | turn=${chatTurns} | intent=${intent} | type=${questionType} | stage2=${stage2Contexts.length > 0}`);
  } finally {
    recordInferenceTime(Date.now() - start);
    releaseInferenceLock();
    releaseServer();
  }
}

// Non-streaming MSR-RAG for the POST route
export async function msrChat(
  sourceId: string | null,
  userMessage: string,
  history: ChatMessage[],
): Promise<string> {
  let result = '';
  for await (const chunk of msrChatStream(sourceId, userMessage, history)) {
    result = chunk;
  }
  return result;
}

function estimateTokens(text: string): number {
  return Math.ceil(text.split(/\s+/).length * 1.33);
}

export function chunkForSummarization(text: string): string[] {
  const cleaned = cleanInputText(text);
  const sentences = cleaned.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 0);

  const MAX_TOKENS = 500;
  const OVERLAP_TOKENS = 50;

  const chunks: string[] = [];
  let current: string[] = [];
  let currentTokens = 0;

  for (const sentence of sentences) {
    const tokens = estimateTokens(sentence);

    if (currentTokens + tokens > MAX_TOKENS && current.length > 0) {
      chunks.push(current.join(' '));

      const overlap: string[] = [];
      let ot = 0;
      for (let i = current.length - 1; i >= 0; i--) {
        const t = estimateTokens(current[i]);
        if (ot + t > OVERLAP_TOKENS) break;
        overlap.unshift(current[i]);
        ot += t;
      }
      current = [...overlap];
      currentTokens = ot;
    }

    current.push(sentence);
    currentTokens += tokens;
  }

  if (current.length > 0) {
    if (chunks.length > 0 && currentTokens < 200) {
      chunks[chunks.length - 1] += ' ' + current.join(' ');
    } else {
      chunks.push(current.join(' '));
    }
  }

  return chunks;
}
