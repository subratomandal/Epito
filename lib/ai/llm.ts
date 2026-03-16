import { extractFocusedContext } from './answer-engine';

const LLAMA_PORT = process.env.LLAMA_SERVER_PORT || '8080';
const LLAMA_URL = `http://127.0.0.1:${LLAMA_PORT}`;
const MODEL = 'mistral-7b-instruct';
const IDLE_TIMEOUT_MS = 120_000; // 2 minutes — unload model after inactivity

console.log(`[LLM] Configured: url=${LLAMA_URL}, model=${MODEL}, idle_timeout=${IDLE_TIMEOUT_MS / 1000}s`);

let inferenceActive = false;
const inferenceQueue: Array<{ resolve: () => void }> = [];

// ─── Idle Model Unloading ────────────────────────────────────────────────────
// After 120s of no AI requests, tell llama-server to unload the model from
// GPU/CPU memory via POST /slots/0?action=erase. The process stays alive;
// the model reloads automatically on the next inference request.
// This matches how Ollama manages model memory.

let idleTimer: ReturnType<typeof setTimeout> | null = null;
let llamaServerRunning = false;

function resetIdleTimer(): void {
  if (idleTimer) clearTimeout(idleTimer);
  llamaServerRunning = true;
  idleTimer = setTimeout(stopIdleLlamaServer, IDLE_TIMEOUT_MS);
}

async function stopIdleLlamaServer(): Promise<void> {
  if (!llamaServerRunning) return;
  // Tell Tauri to kill the llama-server process entirely.
  // This releases ALL memory (model weights + KV cache + GPU VRAM).
  // The process restarts automatically via start_llama_lazy on next AI request.
  try {
    // Call our own API endpoint which invokes the Tauri stop_llama_idle command
    await fetch(`http://127.0.0.1:${process.env.PORT || '3000'}/api/ai/idle-stop`, {
      method: 'POST',
      signal: AbortSignal.timeout(5000),
    });
    llamaServerRunning = false;
    console.log('[LLM] llama-server stopped (idle 120s). Memory released. Restarts on next AI request.');
  } catch {
    // Fallback: at minimum erase the KV cache
    try {
      await fetch(`${LLAMA_URL}/slots/0?action=erase`, { method: 'POST', signal: AbortSignal.timeout(3000) });
      console.log('[LLM] KV cache cleared (idle fallback).');
    } catch {}
  }
}

async function ensureLlamaRunning(): Promise<void> {
  if (llamaServerRunning) return;
  // Check if llama-server is actually responding
  try {
    const res = await fetch(`${LLAMA_URL}/health`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) { llamaServerRunning = true; return; }
  } catch {}
  // Not running — try to start via our API
  console.log('[LLM] llama-server not running, requesting restart...');
  try {
    await fetch(`http://127.0.0.1:${process.env.PORT || '3000'}/api/ai/idle-start`, {
      method: 'POST',
      signal: AbortSignal.timeout(130000), // model loading takes time
    });
    llamaServerRunning = true;
    console.log('[LLM] llama-server restarted successfully.');
  } catch (e) {
    console.error('[LLM] Failed to restart llama-server:', e);
  }
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
  if (!requestedMax) return undefined;
  if (recentInferenceTimes.length < 3 || baselineInferenceMs === 0) return requestedMax;
  const recent = recentInferenceTimes.slice(-3);
  const avgRecent = recent.reduce((a, b) => a + b, 0) / recent.length;
  const ratio = avgRecent / baselineInferenceMs;

  if (ratio > 2.0) return Math.max(100, Math.floor(requestedMax * 0.5));
  if (ratio > 1.5) return Math.max(100, Math.floor(requestedMax * 0.75));
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
  resetIdleTimer();
  const startTime = Date.now();

  try {
    const body: Record<string, unknown> = {
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt },
      ],
      temperature: 0.3,
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
  }
}

async function* callLlamaStream(prompt: string, systemPrompt: string, maxTokens?: number): AsyncGenerator<string> {
  await ensureLlamaRunning();

  const cooldown = computeThermalCooldown();
  if (cooldown > 0) await sleep(cooldown);

  const effectiveMax = adaptiveMaxTokens(maxTokens);

  await acquireInferenceLock();
  resetIdleTimer();
  const startTime = Date.now();

  try {
    const body: Record<string, unknown> = {
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt },
      ],
      temperature: 0.3,
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

const SYSTEM_PROMPT = `You are a world-class analyst and educator. You produce comprehensive, deeply insightful, publication-quality outputs.`;

// ═══════════════════════════════════════════════════════════════════════════════
// Summarization Pipeline — 3-stage, max 1 model call (except map_reduce)
//
// Stage 1: Length-based routing (zero inference)
// Stage 2: TextRank extractive pre-filtering (zero inference, <200ms)
// Stage 3: Abstractive summarization (single model call, streamed)
// ═══════════════════════════════════════════════════════════════════════════════

const SUMMARY_PARAMS = { temperature: 0.15, repeat_penalty: 1.15, top_p: 0.9, top_k: 40 };
const MAP_PARAMS = { temperature: 0.1, repeat_penalty: 1.15, top_p: 0.85, top_k: 30 };

// ─── Stage 1: Length Router ──────────────────────────────────────────────────

type SumStrategy = 'passthrough' | 'direct' | 'extractive' | 'map_reduce';

function routeNote(wordCount: number): SumStrategy {
  if (wordCount <= 300) return 'passthrough';
  if (wordCount <= 1000) return 'direct';
  if (wordCount <= 3000) return 'extractive';
  return 'map_reduce';
}

// ─── Stage 2: TextRank Sentence Extraction (zero inference) ──────────────────

function smartSplitSentences(text: string): string[] {
  // Split on sentence boundaries but NOT on Mr. Dr. e.g. i.e. etc.
  return text
    .replace(/\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\./g, '$1\u0000')
    .split(/(?<=[.!?])\s+/)
    .map(s => s.replace(/\u0000/g, '.').trim())
    .filter(s => s.length > 0);
}

function textRankExtract(text: string, budgetTokens: number): string[] {
  const sentences = smartSplitSentences(text);
  if (sentences.length <= 3) return sentences;

  // Word overlap similarity matrix
  const n = sentences.length;
  const wordSets = sentences.map(s => new Set(s.toLowerCase().split(/\s+/).filter(w => w.length > 3)));
  const sim: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = wordSets[i], b = wordSets[j];
      if (a.size === 0 || b.size === 0) continue;
      let inter = 0;
      for (const w of a) if (b.has(w)) inter++;
      const score = inter / (a.size + b.size - inter);
      if (score > 0.1) { sim[i][j] = score; sim[j][i] = score; }
    }
  }

  // PageRank (20 iterations, damping 0.85)
  let scores = new Array(n).fill(1 / n);
  for (let iter = 0; iter < 20; iter++) {
    const next = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const rowSum = sim[j].reduce((a, b) => a + b, 0);
        if (rowSum > 0) next[i] += 0.85 * (sim[j][i] / rowSum) * scores[j];
      }
      next[i] += 0.15 / n;
    }
    scores = next;
  }

  // Apply scoring boosts
  const paragraphs = text.split(/\n\s*\n/);
  let paragraphStarts = new Set<number>();
  let cursor = 0;
  for (const p of paragraphs) {
    const firstSent = smartSplitSentences(p.trim())[0];
    if (firstSent) {
      const idx = sentences.indexOf(firstSent);
      if (idx >= 0) paragraphStarts.add(idx);
    }
  }

  const NAMED_ENTITY_RE = /[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+/;
  const NUMBER_RE = /\d+/;

  for (let i = 0; i < n; i++) {
    if (i === 0) scores[i] *= 1.3; // First sentence of document
    if (paragraphStarts.has(i)) scores[i] *= 1.15;
    if (NAMED_ENTITY_RE.test(sentences[i])) scores[i] *= 1.1;
    if (NUMBER_RE.test(sentences[i])) scores[i] *= 1.1;
    const wc = sentences[i].split(/\s+/).length;
    if (wc < 5) scores[i] *= 0.5;
    if (wc > 50) scores[i] *= 0.9;
  }

  // Select top sentences within budget, return in original order
  const ranked = scores.map((s, i) => ({ i, s })).sort((a, b) => b.s - a.s);
  const selected = new Set<number>();
  let tokens = 0;
  for (const { i } of ranked) {
    const tk = Math.ceil(sentences[i].split(/\s+/).length * 1.3);
    if (tokens + tk > budgetTokens) continue;
    selected.add(i);
    tokens += tk;
  }

  return sentences.filter((_, i) => selected.has(i));
}

// ─── Stage 3: Abstractive Summarization ──────────────────────────────────────

function validateSummary(summary: string, source: string): string {
  if (!summary || summary.trim().length < 10) return '';

  // Dedup sentences
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

  // Clean encoding artifacts
  let result = deduped.join(' ')
    .replace(/â€™/g, "'").replace(/â€œ/g, '"').replace(/â€/g, '"')
    .replace(/Ã©/g, 'é').replace(/ï¿½/g, '').replace(/Â/g, '');

  return cleanSummaryOutput(result);
}

// ─── Public API (drop-in replacements) ───────────────────────────────────────

export async function summarizeText(text: string): Promise<{ summary: string; keyPoints: string[] } | null> {
  const cleaned = cleanInputText(text);
  const wc = cleaned.split(/\s+/).length;
  const strategy = routeNote(wc);
  console.log(`[Summary] ${wc} words → ${strategy} strategy`);

  if (strategy === 'passthrough') {
    return { summary: cleaned, keyPoints: [] };
  }

  await ensureLlamaRunning();

  if (strategy === 'direct') {
    const prompt = `Summarize the following note concisely. Preserve all key facts, names, dates, and conclusions. Do not add information not present in the note.\n\nNote:\n${truncTk(cleaned, 2800)}\n\nWrite a clear summary in 3-5 sentences:`;
    const response = await callLLM(prompt, SYSTEM_PROMPT, 200);
    const validated = validateSummary(response, cleaned);
    return { summary: validated || cleaned.slice(0, 500), keyPoints: [] };
  }

  if (strategy === 'extractive') {
    const extracted = textRankExtract(cleaned, 600);
    const extractedText = extracted.join(' ');
    console.log(`[Summary] TextRank: ${cleaned.split(/\s+/).length} words → ${extracted.length} sentences (${extractedText.split(/\s+/).length} words)`);

    const prompt = `The following are key sentences extracted from a note. Synthesize them into a clear, coherent summary. Preserve all important facts, names, and conclusions. Do not add information not present below.\n\nKey sentences:\n${extractedText}\n\nWrite a clear summary in 4-6 sentences:`;
    const response = await callLLM(prompt, SYSTEM_PROMPT, 250);
    const validated = validateSummary(response, cleaned);
    return { summary: validated || extractedText, keyPoints: [] };
  }

  // map_reduce
  return mapReduceSummarize(cleaned);
}

export async function* summarizeTextStream(text: string): AsyncGenerator<string> {
  const cleaned = cleanInputText(text);
  const wc = cleaned.split(/\s+/).length;
  const strategy = routeNote(wc);

  if (strategy === 'passthrough') { yield cleaned; return; }

  await ensureLlamaRunning();

  if (strategy === 'direct') {
    const prompt = `Summarize the following note concisely. Preserve all key facts, names, dates, and conclusions.\n\nNote:\n${truncTk(cleaned, 2800)}\n\nWrite a clear summary in 3-5 sentences:`;
    for await (const chunk of streamLLM(prompt, SYSTEM_PROMPT, 200)) {
      yield cleanSummaryOutput(chunk);
    }
    return;
  }

  if (strategy === 'extractive') {
    const extracted = textRankExtract(cleaned, 600);
    const prompt = `Synthesize these key sentences into a clear summary. Preserve all facts and names.\n\nKey sentences:\n${extracted.join(' ')}\n\nWrite a clear summary in 4-6 sentences:`;
    for await (const chunk of streamLLM(prompt, SYSTEM_PROMPT, 250)) {
      yield cleanSummaryOutput(chunk);
    }
    return;
  }

  // map_reduce: stream only the final merge
  const chunkSummaries = await mapPhase(cleaned);
  const mergePrompt = `Merge these section summaries into one coherent summary. Remove redundancy. Preserve all key facts.\n\nSection summaries:\n${chunkSummaries.join('\n\n')}\n\nWrite a unified summary in 4-6 sentences:`;
  for await (const chunk of streamLLM(mergePrompt, SYSTEM_PROMPT, 200)) {
    yield cleanSummaryOutput(chunk);
  }
}

export async function summarizeChunks(chunks: string[]): Promise<{ summary: string; keyPoints: string[] } | null> {
  const combined = chunks.join('\n\n');
  return summarizeText(combined);
}

export async function* summarizeChunksStream(chunks: string[]): AsyncGenerator<string> {
  const combined = chunks.join('\n\n');
  yield* summarizeTextStream(combined);
}

// ─── Map-Reduce Internals ────────────────────────────────────────────────────

async function mapPhase(text: string): Promise<string[]> {
  const sentences = smartSplitSentences(text);
  const chunks: string[] = [];
  let current: string[] = [];
  let currentTk = 0;

  for (const s of sentences) {
    const tk = Math.ceil(s.split(/\s+/).length * 1.3);
    if (currentTk + tk > 700 && current.length > 0) {
      chunks.push(current.join(' '));
      current = [];
      currentTk = 0;
    }
    current.push(s);
    currentTk += tk;
  }
  if (current.length > 0) chunks.push(current.join(' '));

  console.log(`[Summary] Map-reduce: ${chunks.length} chunks`);
  const summaries: string[] = [];

  for (let i = 0; i < chunks.length; i++) {
    const prompt = `Summarize this text in 2-3 sentences. Keep all key facts and names.\n\n${chunks[i]}\n\nSummary:`;
    const res = await callLLM(prompt, SYSTEM_PROMPT, 80);
    summaries.push(cleanSummaryOutput(res));
    // Thermal throttle: 100ms between calls
    await new Promise(r => setTimeout(r, 100));
  }

  return summaries;
}

async function mapReduceSummarize(text: string): Promise<{ summary: string; keyPoints: string[] }> {
  const chunkSummaries = await mapPhase(text);
  const mergePrompt = `Merge these section summaries into one coherent summary. Remove redundancy. Preserve all key facts.\n\nSection summaries:\n${chunkSummaries.join('\n\n')}\n\nWrite a unified summary in 4-6 sentences:`;
  const merged = await callLLM(mergePrompt, SYSTEM_PROMPT, 200);
  return { summary: validateSummary(merged, text) || chunkSummaries.join(' '), keyPoints: [] };
}

function truncTkSum(text: string, max: number): string {
  const w = text.split(/\s+/), m = Math.floor(max / 1.3);
  return w.length <= m ? text : w.slice(0, m).join(' ') + '...';
}

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

// ═══════════════════════════════════════════════════════════════════════════════
// Mistral 7B Chat Pipeline — 8-step architecture for response quality
//
// Pipeline: frustration → correction → intent → rewrite → retrieve →
//           assemble → generate → validate
//
// Fixes: entity fixation, incomplete extraction, context poisoning,
//        coreference failure, correction blindness, frustration spiral
// ═══════════════════════════════════════════════════════════════════════════════

type ChatMessage = { role: string; content: string };
type ChatIntent = 'EXHAUSTIVE_LIST' | 'PERSON_QUERY' | 'FACT_LOOKUP' | 'YES_NO' | 'SUMMARY_REQUEST' | 'COMPARISON' | 'FOLLOW_UP' | 'CORRECTION' | 'TOPIC_CHANGE';

const TOKEN_BUDGET = { system: 250, summary: 100, context: 800, recentChat: 500, query: 150 };
const CHAT_MAX_TOKENS = 250;
const MAX_RECENT = 6; // 3 turns
const KV_RESET_INTERVAL = 6;
const FALLBACK = "I couldn't find a clear answer in your notes. Could you rephrase your question?";

// Session state
let cachedSummary = '';
let cachedSummaryLen = 0;
let chatTurns = 0;
const excludedEntities: Set<string> = new Set();
const recentResponses: string[] = [];

// Two-tier summary: Tier 1 = rolling context (compressed aggressively).
// Tier 2 = pinned facts (persist until explicitly changed by user).
const pinnedFacts: string[] = [];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function countTk(text: string): number { return Math.ceil(text.split(/\s+/).length * 1.3); }
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

// ─── Step 0: Distress Detection (runs BEFORE everything, no model call) ──────
// ISACA 2025 documented lawsuits against AI apps that failed to handle distress.
// This is a legal and ethical requirement, not a feature.

const DISTRESS_RE = /\b(hate my life|want to die|kill myself|end it all|no point in living|suicide|self.?harm|hurt myself|nobody cares|worthless|hopeless|can'?t go on|don'?t want to live)\b/i;

const DISTRESS_RESPONSE = "It sounds like you're going through something really difficult. I'm a notes assistant and not the right resource for this, but please talk to someone who can help.\n\n988 Suicide and Crisis Lifeline: call or text 988 (US)\nCrisis Text Line: text HOME to 741741\n\nYou're not alone, and these feelings can get better with support.";

function detectDistress(msg: string): boolean {
  return DISTRESS_RE.test(msg);
}

// ─── Step 1: Frustration Detection ───────────────────────────────────────────

const FRUSTRATION_RE = /\b(dumb|stupid|wtf|idiot|useless|wrong again|i already said|i told you|i just asked|for the .* time|forget it|never mind|are you even reading|not what i asked|that's not it|no that's wrong|you're wrong|still wrong)\b/i;

function detectFrustration(msg: string): boolean {
  return FRUSTRATION_RE.test(msg);
}

// ─── Step 2: Correction Detection ────────────────────────────────────────────

async function detectCorrection(msg: string, lastResponse: string): Promise<{ isCorrection: boolean; assertion: string }> {
  if (!lastResponse) return { isCorrection: false, assertion: '' };

  const answer = await quickCall(
    `Is the user saying the previous answer was wrong or incomplete? Answer only YES or NO.\n\nPrevious answer: ${lastResponse.slice(0, 150)}\nUser message: ${msg}`
  );

  if (answer.toUpperCase().startsWith('YES')) {
    // Extract what the user is asserting as correct
    const corrected = msg.replace(/\b(no|wrong|incorrect|not|that's not|you're wrong)\b/gi, '').trim();
    return { isCorrection: true, assertion: corrected || msg };
  }
  return { isCorrection: false, assertion: '' };
}

// ─── Step 3: Intent Classification ───────────────────────────────────────────

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

// ─── Step 4: Query Rewriting ─────────────────────────────────────────────────

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

async function decomposeQuery(query: string): Promise<string[]> {
  const result = await quickCall(
    `The user wants a complete list. Generate 3 different search queries that would find all instances from different parts of a document. Each query should use different keywords.\n\nOriginal: ${query}\n\nOutput 3 queries, one per line:`,
    80
  );
  const variants = result.split('\n').map(l => l.replace(/^\d+[\.\)]\s*/, '').trim()).filter(l => l.length > 3);
  return variants.length > 0 ? [query, ...variants.slice(0, 3)] : [query];
}

// ─── Step 5: Context Compression ─────────────────────────────────────────────

// ─── P6: Context Sufficiency Gate ────────────────────────────────────────────
// Google ICLR 2025: insufficient context makes hallucination WORSE.
// If retrieved chunks score below threshold, force abstention instead of
// feeding irrelevant context that increases model confidence in wrong answers.

type ContextSufficiency = 'SUFFICIENT' | 'LOW' | 'INSUFFICIENT' | 'NO_CONTEXT';

function checkContextSufficiency(chunks: string[], avgScore: number): ContextSufficiency {
  if (chunks.length === 0) return 'NO_CONTEXT';
  if (avgScore < 0.25) return 'INSUFFICIENT';
  if (avgScore < 0.45) return 'LOW';
  return 'SUFFICIENT';
}

// ─── P13: Negation Handling (code-based, no model) ───────────────────────────
// 7B models fail at negation. "Which are NOT mentioned" returns the mentioned ones.
// Fix: detect negation, decompose into positive extraction, compute complement in code.

const NEGATION_RE = /\b(not|n'?t|never|no|none|neither|nor|except|other than|besides|excluding|without)\b.*\b(mention|include|list|appear|use|contain|have|reference)\b/i;

function hasNegation(query: string): boolean {
  return NEGATION_RE.test(query);
}

// ─── P14: Numerical Query Detection (code computes, not model) ───────────────

const NUMERICAL_RE = /\b(average|mean|sum|total|count|how many|percentage|ratio|maximum|minimum|max|min|fastest|slowest|highest|lowest)\b/i;

function isNumericalQuery(query: string): boolean {
  return NUMERICAL_RE.test(query);
}

function extractNumbersFromText(text: string): number[] {
  const matches = text.match(/\b\d[\d,.]*\b/g) || [];
  return matches.map(m => parseFloat(m.replace(/,/g, ''))).filter(n => !isNaN(n) && isFinite(n));
}

function computeNumerical(query: string, numbers: number[]): string | null {
  if (numbers.length === 0) return null;
  const q = query.toLowerCase();

  if (/\b(average|mean)\b/.test(q)) {
    const avg = numbers.reduce((a, b) => a + b, 0) / numbers.length;
    return `The computed average is ${avg.toFixed(2)} (from ${numbers.length} values: ${numbers.join(', ')}).`;
  }
  if (/\b(sum|total)\b/.test(q)) {
    return `The total is ${numbers.reduce((a, b) => a + b, 0).toFixed(2)}.`;
  }
  if (/\b(count|how many)\b/.test(q)) {
    return `Count: ${numbers.length}.`;
  }
  if (/\b(max|maximum|highest|fastest)\b/.test(q)) {
    return `The maximum value is ${Math.max(...numbers)}.`;
  }
  if (/\b(min|minimum|lowest|slowest)\b/.test(q)) {
    return `The minimum value is ${Math.min(...numbers)}.`;
  }
  return null;
}

function compressChunks(chunks: string[], maxTokens: number, excludeEntities: Set<string>): string {
  if (chunks.length === 0) return '';
  const compressed: string[] = [];
  let total = 0;

  for (let i = 0; i < chunks.length && i < 5; i++) {
    let chunk = chunks[i];

    // Deprioritize chunks primarily about excluded entities
    if (excludeEntities.size > 0) {
      const lower = chunk.toLowerCase();
      const excluded = [...excludeEntities].some(e => {
        const re = new RegExp(`\\b${e.toLowerCase()}\\b`, 'g');
        return (lower.match(re) || []).length > 2;
      });
      if (excluded) continue;
    }

    chunk = chunk.replace(/\b(however|moreover|furthermore|additionally|it is worth noting that)\b/gi, '').replace(/\s{2,}/g, ' ').trim();
    const sentences = chunk.split(/(?<=[.!?])\s+/);
    if (sentences.length > 3) chunk = sentences.slice(0, 3).join(' ');

    const tk = countTk(chunk);
    if (total + tk > maxTokens) {
      const rem = maxTokens - total;
      if (rem > 30) compressed.push(`[${compressed.length + 1}] ${truncTk(chunk, rem)}`);
      break;
    }
    compressed.push(`[${compressed.length + 1}] ${chunk}`);
    total += tk;
  }
  return compressed.join('\n\n');
}

// ─── Step 6: Summary Management ──────────────────────────────────────────────

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

// ─── Step 7: Prompt Assembly & Response Design ──────────────────────────────
// System prompt: exactly 6 rules, <200 tokens. Each rule prevents a documented failure.

// System prompt: 3 rules, <50 tokens. Research (P10) shows Mistral ignores more.
const BASE_SYSTEM = `You are Epito, a note assistant. Answer using only the provided notes. If the answer is not in the notes, say "I don't have that in your notes." Be specific and cite the note name.`;

const GROUNDING = `Answer using only the notes above. If the information is not there, say you don't have it. When stating a fact, cite the note name in parentheses.`;

// Intent-specific prompt additions (one-liner per intent, appended to user query)
const INTENT_PROMPTS: Record<ChatIntent, string> = {
  FACT_LOOKUP: 'Answer in one sentence. State the fact and which note it comes from.',
  YES_NO: 'Answer YES or NO first, then give one sentence of evidence from the notes.',
  EXHAUSTIVE_LIST: 'List every instance found in the notes. One per line. Include the source note. Do not stop until all instances are listed.',
  PERSON_QUERY: 'Describe this person based only on what the notes say. List their specific contributions or roles mentioned. Cite the source note for each fact.',
  COMPARISON: 'Compare using specific data from the notes. If the notes contain numbers, use them. State which is better and why based on the data.',
  SUMMARY_REQUEST: 'Give a concise overview based on the notes. Cover the main points in 3-4 sentences.',
  FOLLOW_UP: '',
  CORRECTION: 'The user corrected your previous answer. Acknowledge this and provide the right information using the notes.',
  TOPIC_CHANGE: '',
};

const INTENT_MAX_TOKENS: Record<ChatIntent, number> = {
  FACT_LOOKUP: 60,
  YES_NO: 40,
  EXHAUSTIVE_LIST: 200,
  PERSON_QUERY: 150,
  COMPARISON: 200,
  SUMMARY_REQUEST: 150,
  FOLLOW_UP: 150,
  CORRECTION: 150,
  TOPIC_CHANGE: 150,
};

function buildMessages(
  context: string, summary: string, recent: ChatMessage[], query: string,
  flags: { frustrated: boolean; isCorrection: boolean; intent: ChatIntent },
): Array<{ role: string; content: string }> {
  let sys = BASE_SYSTEM;

  if (flags.frustrated) {
    sys = `Give a direct answer to the user's question. Do not apologize. Do not reference previous answers. Just answer correctly.\n\n${sys}`;
  }
  if (flags.isCorrection) {
    // Rule already in INTENT_PROMPTS.CORRECTION, no need to bloat system prompt
  }
  for (const e of excludedEntities) {
    sys += `\nDo not mention ${e} unless the user specifically asks about them.`;
  }

  sys = truncTk(sys, TOKEN_BUDGET.system);

  if (summary) {
    sys += `\n\n### Previous Context (background only, NOT the current topic)\nThe following is background from earlier. Do not reference it unless asked.\n${summary}`;
  }

  // Relevant Notes: explicit "no results" when empty (prevents hallucination from training data)
  if (context && context.trim()) {
    sys += `\n\n### Relevant Notes\n${context}`;
  } else {
    sys += `\n\n### Relevant Notes\nNo matching notes found for this question.`;
  }

  const msgs: Array<{ role: string; content: string }> = [{ role: 'system', content: sys }];
  for (const m of recent) msgs.push({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content });

  // User query with intent-specific instruction + grounding anchor
  const intentPrompt = INTENT_PROMPTS[flags.intent] || '';
  const queryWithGrounding = intentPrompt
    ? `${query}\n\n${intentPrompt}\n\n${GROUNDING}`
    : `${query}\n\n${GROUNDING}`;
  msgs.push({ role: 'user', content: queryWithGrounding });

  return msgs;
}

// ─── Step 8: Output Validation ───────────────────────────────────────────────

// ─── Grounding Check (post-generation entity validation) ─────────────────────

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

function detectLoop(text: string): boolean {
  const words = text.toLowerCase().split(/\s+/);
  if (words.length < 24) return false;
  const ngrams = new Map<string, number>();
  for (let i = 0; i <= words.length - 8; i++) {
    const g = words.slice(i, i + 8).join(' ');
    ngrams.set(g, (ngrams.get(g) || 0) + 1);
    if ((ngrams.get(g) || 0) >= 3) return true;
  }
  return false;
}

function validateOutput(text: string, query: string, context: string): 'ok' | 'empty' | 'repetition' | 'off-topic' | 'excluded-entity' {
  if (!text || text.trim().length < 5) return 'empty';
  if (detectLoop(text)) return 'repetition';

  const sentences = text.split(/[.!?]+/).map(s => s.trim().toLowerCase()).filter(s => s.length > 15);
  const seen = new Set<string>();
  for (const s of sentences) { if (seen.has(s)) return 'repetition'; seen.add(s); }

  // Check excluded entity violation
  for (const entity of excludedEntities) {
    if (text.toLowerCase().includes(entity.toLowerCase()) && !query.toLowerCase().includes(entity.toLowerCase())) {
      return 'excluded-entity';
    }
  }

  // Check similarity to recent responses (entity fixation detection)
  for (const prev of recentResponses.slice(-3)) {
    const prevWords = new Set(prev.toLowerCase().split(/\s+/).filter(w => w.length > 4));
    const curWords = text.toLowerCase().split(/\s+/).filter(w => w.length > 4);
    const overlap = curWords.filter(w => prevWords.has(w));
    if (overlap.length > curWords.length * 0.7 && curWords.length > 10) return 'repetition';
  }

  if (query && context) {
    const qw = new Set(query.toLowerCase().split(/\s+/).filter(w => w.length > 4));
    const cw = new Set(context.toLowerCase().split(/\s+/).filter(w => w.length > 4).slice(0, 100));
    const rw = text.toLowerCase().split(/\s+/).filter(w => w.length > 4);
    if (rw.filter(w => qw.has(w) || cw.has(w)).length === 0 && rw.length > 10) return 'off-topic';
  }

  return 'ok';
}

// ─── Core Pipeline ───────────────────────────────────────────────────────────

function getParams(frustrated: boolean, isCorrection: boolean, intent: ChatIntent) {
  const maxTk = INTENT_MAX_TOKENS[intent] || CHAT_MAX_TOKENS;
  if (frustrated) return { temperature: 0.1, repeat_penalty: 1.25, presence_penalty: 0.2, top_p: 0.85, top_k: 30, max_tokens: maxTk };
  if (isCorrection) return { temperature: 0.15, repeat_penalty: 1.25, presence_penalty: 0.2, top_p: 0.85, top_k: 30, max_tokens: maxTk };
  return { temperature: 0.2, repeat_penalty: 1.2, presence_penalty: 0.15, top_p: 0.9, top_k: 40, max_tokens: maxTk };
}

async function chatPipeline(
  rawContext: string,
  rawQuery: string,
  history: ChatMessage[],
  isRAG: boolean,
  chunks?: string[],
): Promise<string> {
  await ensureLlamaRunning();
  await acquireInferenceLock();
  resetIdleTimer();
  const start = Date.now();

  try {
    chatTurns++;

    // Step 0: Distress detection (BEFORE everything, no model call, no retrieval)
    if (detectDistress(rawQuery)) {
      console.log('[Chat] DISTRESS detected — returning crisis response');
      return DISTRESS_RESPONSE;
    }

    const lastResponse = history.length > 0 ? history[history.length - 1]?.content || '' : '';

    // Step 1: Frustration detection
    const frustrated = detectFrustration(rawQuery);
    if (frustrated) console.log('[Chat] Frustration detected');

    // Step 2: Correction detection
    const { isCorrection, assertion } = await detectCorrection(rawQuery, lastResponse);
    if (isCorrection) console.log(`[Chat] Correction detected: "${assertion}"`);

    // Step 3: Intent classification
    const intent = classifyIntent(rawQuery, isCorrection);
    console.log(`[Chat] Intent: ${intent}`);

    // Handle entity exclusion
    const forgetMatch = rawQuery.match(/\b(?:forget|stop talking about|ignore)\s+(.+)/i);
    if (forgetMatch) {
      excludedEntities.add(forgetMatch[1].trim());
      console.log(`[Chat] Excluded entity: "${forgetMatch[1].trim()}"`);
    }

    // KV cache reset
    if (chatTurns % KV_RESET_INTERVAL === 0) {
      console.log(`[Chat] KV cache reset at turn ${chatTurns}`);
      try { await fetch(`${LLAMA_URL}/slots/0?action=erase`, { method: 'POST', signal: AbortSignal.timeout(2000) }); } catch {}
      cachedSummary = '';
      cachedSummaryLen = 0;
    }

    // Step 4: Query rewriting
    const { summary, recent } = await manageSummary(history, intent, frustrated);
    let rewritten = await rewriteQuery(rawQuery, recent);
    if (isCorrection && assertion) rewritten += ` (Correction: ${assertion})`;

    // Step 5: Context preparation
    let context: string;
    if (isRAG && chunks) {
      if (intent === 'EXHAUSTIVE_LIST') {
        // Multi-query: use all chunks, compress less aggressively
        context = compressChunks(chunks, TOKEN_BUDGET.context, excludedEntities);
      } else {
        context = compressChunks(chunks, TOKEN_BUDGET.context, excludedEntities);
      }
    } else {
      context = truncTk(rawContext, TOKEN_BUDGET.context);
    }

    // ── Focused Context Extraction ──────────────────────────────────────
    // Instead of sending full chunks (which 7B models ignore 85% of the time),
    // find query terms in chunks and extract 100 words around each match.
    // The model gets ONLY the relevant surrounding text.
    const allChunks = isRAG && chunks ? chunks : context ? [context] : [];
    const extraction = extractFocusedContext(rawQuery, allChunks);

    // If extraction found a direct answer (e.g. for list queries), return it
    if (extraction.directAnswer) {
      console.log(`[Chat] Direct code answer (${extraction.matchCount} matches)`);
      return extraction.directAnswer;
    }

    // Use focused context (term matches + surrounding words) instead of full chunks
    const effectiveContext = extraction.matchCount > 0
      ? extraction.focusedContext
      : context; // fall back to full context if no terms matched

    // Step 6: Prompt assembly
    const flags = { frustrated, isCorrection, intent };
    let messages: Array<{ role: string; content: string }>;

    if (intent === 'EXHAUSTIVE_LIST' && effectiveContext) {
      const entityType = rawQuery.replace(/\b(list|all|every|mention|name)\b/gi, '').trim() || 'items';
      messages = [{
        role: 'system',
        content: `Extract every ${entityType} mentioned in the following text. Return ONLY the names, one per line. Do not skip any. Do not add explanations.`,
      }, {
        role: 'user',
        content: `Text:\n${effectiveContext}\n\nList:`,
      }];
    } else {
      messages = buildMessages(effectiveContext, summary, recent, rewritten, flags);
    }

    // Step 7: Generation
    const params = getParams(frustrated, isCorrection, intent);
    const body = { messages, ...params, stream: false };

    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`LLM error: ${res.status}`);
    const data = await res.json();
    let response = (data.choices?.[0]?.message?.content || '').trim();

    // Step 8: Output validation
    const validation = validateOutput(response, rewritten, context);
    if (validation !== 'ok') {
      console.warn(`[Chat] Validation: ${validation}. Retrying...`);
      const retryBody = { ...body, temperature: Math.min(params.temperature + 0.15, 0.5), repeat_penalty: 1.25 };
      if (validation === 'repetition' || validation === 'excluded-entity') {
        // Wipe summary to break fixation
        retryBody.messages = buildMessages(context, '', recent, rewritten, { ...flags, frustrated: true });
      }
      const rr = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(retryBody),
      });
      if (rr.ok) {
        const rd = await rr.json();
        const rsp = (rd.choices?.[0]?.message?.content || '').trim();
        response = validateOutput(rsp, rewritten, context) === 'ok' ? rsp : FALLBACK;
      } else {
        response = FALLBACK;
      }
    }

    // Grounding check: flag ungrounded entities/numbers
    const ungrounded = groundingCheck(response, context);
    if (ungrounded.length > 2) {
      console.warn(`[Chat] Grounding: ${ungrounded.length} ungrounded claims: ${ungrounded.join(', ')}`);
      // Regenerate with stricter grounding
      const strictBody = { ...body, temperature: 0.1, max_tokens: params.max_tokens };
      strictBody.messages = buildMessages(context, '', recent, rewritten, { ...flags, frustrated: true });
      try {
        const sr = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(strictBody),
        });
        if (sr.ok) {
          const sd = await sr.json();
          const strict = (sd.choices?.[0]?.message?.content || '').trim();
          if (strict && groundingCheck(strict, context).length < ungrounded.length) {
            response = strict;
          }
        }
      } catch {}
    } else if (ungrounded.length > 0) {
      console.log(`[Chat] Minor grounding gaps: ${ungrounded.join(', ')}`);
    }

    // Track for fixation detection
    recentResponses.push(response);
    if (recentResponses.length > 5) recentResponses.shift();

    console.log(`[Chat] ${Date.now() - start}ms | turn=${chatTurns} | intent=${intent} | frustrated=${frustrated} | correction=${isCorrection}`);
    return response;
  } finally {
    recordInferenceTime(Date.now() - start);
    releaseInferenceLock();
  }
}

// ─── Streaming Pipeline ──────────────────────────────────────────────────────

async function* streamPipeline(
  rawContext: string,
  rawQuery: string,
  history: ChatMessage[],
  isRAG: boolean,
  chunks?: string[],
): AsyncGenerator<string> {
  await ensureLlamaRunning();
  await acquireInferenceLock();
  resetIdleTimer();
  const start = Date.now();

  try {
    chatTurns++;

    // Step 0: Distress detection
    if (detectDistress(rawQuery)) {
      yield DISTRESS_RESPONSE;
      return;
    }

    const lastResponse = history.length > 0 ? history[history.length - 1]?.content || '' : '';

    const frustrated = detectFrustration(rawQuery);
    const { isCorrection, assertion } = await detectCorrection(rawQuery, lastResponse);
    const intent = classifyIntent(rawQuery, isCorrection);

    const forgetMatch = rawQuery.match(/\b(?:forget|stop talking about|ignore)\s+(.+)/i);
    if (forgetMatch) excludedEntities.add(forgetMatch[1].trim());

    if (chatTurns % KV_RESET_INTERVAL === 0) {
      try { await fetch(`${LLAMA_URL}/slots/0?action=erase`, { method: 'POST', signal: AbortSignal.timeout(2000) }); } catch {}
      cachedSummary = ''; cachedSummaryLen = 0;
    }

    const { summary, recent } = await manageSummary(history, intent, frustrated);
    let rewritten = await rewriteQuery(rawQuery, recent);
    if (isCorrection && assertion) rewritten += ` (Correction: ${assertion})`;

    let context = isRAG && chunks ? compressChunks(chunks, TOKEN_BUDGET.context, excludedEntities) : truncTk(rawContext, TOKEN_BUDGET.context);

    // Focused context extraction (same as sync pipeline)
    const allChunksS = isRAG && chunks ? chunks : context ? [context] : [];
    const extractionS = extractFocusedContext(rawQuery, allChunksS);

    if (extractionS.directAnswer) {
      yield extractionS.directAnswer;
      return;
    }

    if (extractionS.matchCount > 0) {
      context = extractionS.focusedContext;
    }

    const flags = { frustrated, isCorrection, intent };
    const messages = intent === 'EXHAUSTIVE_LIST' && context
      ? [{ role: 'system', content: `Extract every item mentioned in the text. Return names only, one per line.` }, { role: 'user', content: `Text:\n${context}\n\nList:` }]
      : buildMessages(context, summary, recent, rewritten, flags);

    const params = getParams(frustrated, isCorrection, intent);
    const body = { messages, ...params, stream: true };

    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
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
            if (detectLoop(accumulated)) {
              const w = accumulated.split(/\s+/);
              accumulated = w.slice(0, Math.max(w.length - 16, Math.floor(w.length * 0.7))).join(' ');
              yield accumulated; return;
            }
            yield accumulated;
          }
        } catch {}
      }
    }

    recentResponses.push(accumulated);
    if (recentResponses.length > 5) recentResponses.shift();
  } finally {
    recordInferenceTime(Date.now() - start);
    releaseInferenceLock();
  }
}

// ─── Public Chat API ─────────────────────────────────────────────────────────

export async function chatWithContext(documentText: string, userMessage: string, history: ChatMessage[]): Promise<string> {
  return chatPipeline(cleanInputText(documentText), userMessage, history, false);
}

export async function* chatWithContextStream(documentText: string, userMessage: string, history: ChatMessage[]): AsyncGenerator<string> {
  yield* streamPipeline(cleanInputText(documentText), userMessage, history, false);
}

export async function chatWithRAG(chunks: string[], userMessage: string, history: ChatMessage[]): Promise<string> {
  return chatPipeline('', userMessage, history, true, chunks);
}

export async function* chatWithRAGStream(chunks: string[], userMessage: string, history: ChatMessage[]): AsyncGenerator<string> {
  yield* streamPipeline('', userMessage, history, true, chunks);
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

const SECTION_SUMMARY_PROMPT = (sectionText: string, index: number, total: number, previousPoints: string) => {
  let prompt = `You are analyzing section ${index + 1} of ${total} from a document.\n\n`;

  if (previousPoints) {
    prompt += `PREVIOUSLY COVERED POINTS (DO NOT REPEAT ANY OF THESE):\n${previousPoints}\n\n`;
  }

  prompt += `Analyze ONLY this section and extract NEW insights not already covered above.

STRICT LIMIT: Your ENTIRE response must be under 150 words.

Output using EXACTLY these section headers where applicable:

KEY IDEAS
- [insight]

IMPORTANT DETAILS
- [detail]

CONCEPTS
- [concept or term worth noting]

Rules:
- Keep total response under 150 words
- Skip any point already covered in previous sections
- Extract insights — do not paraphrase or copy the text
- Each point must be one concise line
- Only include a header if it has at least one point
- Do NOT add meta-commentary, introductions, or conclusions

Section text:
"""
${sectionText}
"""`;

  return prompt;
};

const MERGE_SECTIONS_PROMPT = (allSections: string) =>
`Combine these section summaries into one final coherent summary.

Rules:
- Merge overlapping or similar points into single clear statements
- Remove all redundancy
- Group under these EXACT headers: KEY IDEAS, IMPORTANT DETAILS, CONCEPTS
- Order points by importance within each group
- Keep each point to one concise line
- Preserve all unique information — do not drop non-redundant points

Section summaries:
"""
${allSections}
"""

Output using EXACTLY these headers:

KEY IDEAS
- ...

IMPORTANT DETAILS
- ...

CONCEPTS
- ...`;

export async function* summarizeSectionStream(
  sectionText: string,
  sectionIndex: number,
  totalSections: number,
  previousPoints: string,
): AsyncGenerator<string> {
  const prompt = SECTION_SUMMARY_PROMPT(sectionText, sectionIndex, totalSections, previousPoints);
  for await (const chunk of streamLLM(prompt, SYSTEM_PROMPT, MAX_OUTPUT_TOKENS)) {
    yield cleanSummaryOutput(chunk);
  }
}

export async function* mergeSectionsStream(
  sectionSummaries: string[],
): AsyncGenerator<string> {
  const all = sectionSummaries.map((s, i) => `[Section ${i + 1}]\n${s}`).join('\n\n');
  for await (const chunk of streamLLM(MERGE_SECTIONS_PROMPT(all), SYSTEM_PROMPT)) {
    yield cleanSummaryOutput(chunk);
  }
}
