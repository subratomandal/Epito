import { extractFocusedContext } from './answer-engine';
import { generateEmbedding } from '@/memory/embeddings';
import { cosineSimilarity } from '@/memory/vector';
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

const SUMMARY_PARAMS = { temperature: 0.6, repeat_penalty: 1.1, top_p: 0.9, top_k: 50 };

// --- Length Router

type SumStrategy = 'passthrough' | 'direct' | 'hierarchical';

function routeNote(wordCount: number): SumStrategy {
  if (wordCount <= 300) return 'passthrough';
  if (wordCount <= 800) return 'direct';
  return 'hierarchical';
}

// --- Sentence Segmentation

function smartSplitSentences(text: string): string[] {
  return text
    .replace(/\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\./g, '$1\u0000')
    .split(/(?<=[.!?])\s+/)
    .map(s => s.replace(/\u0000/g, '.').trim())
    .filter(s => s.length > 0);
}

// --- Two-Phase Sentence Ranking
// Phase 1: TextRank (zero model calls) -> top 60. Phase 2: embed only those for clustering.

function textRankSelect(sentences: string[], topN: number): { sentence: string; index: number; score: number }[] {
  const n = sentences.length;
  if (n <= topN) return sentences.map((s, i) => ({ sentence: s, index: i, score: 1 }));

  // Word overlap similarity matrix (zero model calls)
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

  // Boost entity-containing and first sentences
  const ENTITY_RE = /[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+/;
  const NUMBER_RE = /\d+/;
  for (let i = 0; i < n; i++) {
    if (i === 0) scores[i] *= 1.3;
    if (ENTITY_RE.test(sentences[i])) scores[i] *= 1.15;
    if (NUMBER_RE.test(sentences[i])) scores[i] *= 1.1;
    const wc = sentences[i].split(/\s+/).length;
    if (wc < 5) scores[i] *= 0.5;
  }

  const ranked = scores.map((s, i) => ({ sentence: sentences[i], index: i, score: s }));
  ranked.sort((a, b) => b.score - a.score);
  return ranked.slice(0, topN);
}

async function embedSelectedSentences(
  selected: { sentence: string; index: number; score: number }[],
): Promise<{ sentence: string; index: number; score: number; embedding: number[] }[]> {
  // Embed ONLY the pre-selected sentences (40-60, not all 150+)
  const results: { sentence: string; index: number; score: number; embedding: number[] }[] = [];
  for (const item of selected) {
    const emb = await generateEmbedding(item.sentence);
    results.push({ ...item, embedding: emb });
  }
  return results;
}

// --- K-Means Topic Clustering
// Groups sentences into topic clusters for full-document coverage.

function kMeansCluster(
  items: { embedding: number[]; sentence: string; index: number }[],
  k: number,
  maxIter = 20,
): { sentence: string; index: number }[][] {
  if (items.length <= k) return items.map(item => [item]);

  const dim = items[0].embedding.length;

  // Initialize centroids using k-means++ (spread-out initialization)
  const centroids: number[][] = [];
  centroids.push([...items[0].embedding]);
  for (let c = 1; c < k; c++) {
    let maxDist = -1;
    let bestIdx = 0;
    for (let i = 0; i < items.length; i++) {
      let minDist = Infinity;
      for (const cent of centroids) {
        const d = 1 - cosineSimilarity(items[i].embedding, cent);
        if (d < minDist) minDist = d;
      }
      if (minDist > maxDist) { maxDist = minDist; bestIdx = i; }
    }
    centroids.push([...items[bestIdx].embedding]);
  }

  // Iterate
  let assignments = new Array(items.length).fill(0);
  for (let iter = 0; iter < maxIter; iter++) {
    // Assign each item to nearest centroid
    const newAssignments = items.map((item, i) => {
      let bestCluster = 0;
      let bestSim = -Infinity;
      for (let c = 0; c < k; c++) {
        const sim = cosineSimilarity(item.embedding, centroids[c]);
        if (sim > bestSim) { bestSim = sim; bestCluster = c; }
      }
      return bestCluster;
    });

    // Check convergence
    if (newAssignments.every((a, i) => a === assignments[i])) break;
    assignments = newAssignments;

    // Update centroids
    for (let c = 0; c < k; c++) {
      const members = items.filter((_, i) => assignments[i] === c);
      if (members.length === 0) continue;
      for (let d = 0; d < dim; d++) {
        centroids[c][d] = members.reduce((sum, m) => sum + m.embedding[d], 0) / members.length;
      }
    }
  }

  // Group items by cluster, sorted by original document order within each
  const clusters: { sentence: string; index: number }[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < items.length; i++) {
    clusters[assignments[i]].push({ sentence: items[i].sentence, index: items[i].index });
  }
  // Sort within each cluster by document order
  for (const cluster of clusters) cluster.sort((a, b) => a.index - b.index);
  // Remove empty clusters and sort clusters by earliest sentence
  return clusters.filter(c => c.length > 0).sort((a, b) => a[0].index - b[0].index);
}

// --- Summary Validation

function validateSummary(summary: string, source: string): string {
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

async function extractiveAbstractivePipeline(text: string): Promise<{
  clusterSummaries: string[];
  entityContext: string;
}> {
  // Step 1: Sentence segmentation
  const sentences = smartSplitSentences(text);
  console.log(`[Summary] Pipeline: ${sentences.length} sentences`);

  // Step 2a: TextRank selection (ZERO model calls, ~50ms)
  const topN = Math.min(50, sentences.length);
  const textRankSelected = textRankSelect(sentences, topN);
  console.log(`[Summary] TextRank: ${sentences.length} → ${textRankSelected.length} top sentences (zero compute)`);

  // Step 2b: Embed ONLY the selected sentences for clustering (~40 embeddings, not 150+)
  const ranked = await embedSelectedSentences(textRankSelected);
  console.log(`[Summary] Embedded ${ranked.length} sentences for clustering`);

  // Step 3: Topic clustering — group into 4-8 topic clusters
  const k = Math.min(Math.max(4, Math.ceil(ranked.length / 6)), 8);
  const clusters = kMeansCluster(ranked, k);
  console.log(`[Summary] K-means: ${ranked.length} sentences → ${clusters.length} topic clusters`);

  // Step 4: Per-cluster LLM summaries
  const clusterSummaries: string[] = [];
  for (let i = 0; i < clusters.length; i++) {
    const clusterText = clusters[i].map(s => s.sentence).join(' ');
    const prompt = `Summarize the following text in 3-5 sentences. Preserve ALL names of people, organizations, institutions, dates, and technologies. Explain relationships between entities.\n\nText:\n${clusterText}\n\nSummary:`;
    const res = await callLLM(prompt, SYSTEM_PROMPT, 200);
    clusterSummaries.push(cleanSummaryOutput(res));
    if (i < clusters.length - 1) await new Promise(r => setTimeout(r, 150));
  }

  // Step 5: Entity extraction from all cluster summaries
  const allSummaryText = clusterSummaries.join(' ');
  const entities: string[] = [];
  const orgRe = /[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|and|for))*\s+(?:University|Institute|College|Academy|Inc|Corp|Ltd|Foundation)/g;
  const personRe = /[A-Z][a-z]{1,15}\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{1,15}/g;
  let m;
  while ((m = orgRe.exec(allSummaryText)) !== null) entities.push(m[0]);
  while ((m = personRe.exec(allSummaryText)) !== null) entities.push(m[0]);
  const uniqueEntities = [...new Set(entities)];
  const entityContext = uniqueEntities.length > 0
    ? `Key entities mentioned: ${uniqueEntities.slice(0, 20).join(', ')}`
    : '';

  console.log(`[Summary] ${clusters.length} cluster summaries, ${uniqueEntities.length} entities extracted`);
  return { clusterSummaries, entityContext };
}

function buildGlobalSynthesisPrompt(clusterSummaries: string[], entityContext: string): string {
  const sections = clusterSummaries.map((s, i) => `[Topic ${i + 1}] ${s}`).join('\n\n');
  let prompt = `You are writing a comprehensive summary of an entire document.\n\n`;
  if (entityContext) prompt += `${entityContext}\n\n`;
  prompt += `Topic summaries from different parts of the document:\n${sections}\n\n`;
  prompt += `Write a detailed, coherent summary (aim for 400-800 words) that:\n`;
  prompt += `- Explains the overall topic and purpose of the document\n`;
  prompt += `- Identifies ALL key people, organizations, and their relationships\n`;
  prompt += `- Covers every topic above — do not skip any section\n`;
  prompt += `- Explains the meaning and significance, not just facts\n`;
  prompt += `- Does NOT repeat the same information twice\n`;
  prompt += `- Reads as a single coherent narrative, not a list of bullet points\n\nSummary:`;
  return prompt;
}

async function hierarchicalSummarize(text: string): Promise<{ summary: string; keyPoints: string[] }> {
  const { clusterSummaries, entityContext } = await extractiveAbstractivePipeline(text);
  const globalPrompt = buildGlobalSynthesisPrompt(clusterSummaries, entityContext);
  const result = await callLLM(globalPrompt, SYSTEM_PROMPT, 1300);
  const validated = validateSummary(result, text);
  return {
    summary: validated || clusterSummaries.join(' '),
    keyPoints: [],
  };
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

// --- Mistral 7B Chat Pipeline (8-step)
// frustration -> correction -> intent -> rewrite -> retrieve -> assemble -> generate -> validate

type ChatMessage = { role: string; content: string };
type ChatIntent = 'EXHAUSTIVE_LIST' | 'PERSON_QUERY' | 'FACT_LOOKUP' | 'YES_NO' | 'SUMMARY_REQUEST' | 'COMPARISON' | 'FOLLOW_UP' | 'CORRECTION' | 'TOPIC_CHANGE';

const TOKEN_BUDGET = { system: 250, summary: 100, context: 1400, recentChat: 400, query: 150 };
const CHAT_MAX_TOKENS = 1024;
const MAX_RECENT = 6; // 3 turns
const KV_RESET_INTERVAL = 6;
const FALLBACK = "I couldn't find an answer in your notes for this.";

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

function buildEntityContext(query: string): string {
  try {
    // Get all entities from the database for any relevant documents
    const allEntities = db.searchEntities('%', undefined);
    if (allEntities.length === 0) return '';

    // Group entities by type
    const byType = new Map<string, Set<string>>();
    const entityDocs = new Map<string, string>(); // entity → document_id

    for (const e of allEntities) {
      const type = e.entity_type;
      if (!byType.has(type)) byType.set(type, new Set());
      byType.get(type)!.add(e.entity_name);
      entityDocs.set(e.entity_name.toLowerCase(), e.document_id);
    }

    // Build relationship lines
    const lines: string[] = [];

    const orgs = byType.get('ORG');
    const people = byType.get('PERSON');
    const tools = byType.get('TOOL');

    // Find co-occurring entities (same document = likely related)
    if (people && people.size > 0 && orgs && orgs.size > 0) {
      for (const person of people) {
        const personDoc = entityDocs.get(person.toLowerCase());
        if (!personDoc) continue;
        const relatedOrgs: string[] = [];
        for (const org of orgs) {
          if (entityDocs.get(org.toLowerCase()) === personDoc) {
            relatedOrgs.push(org);
          }
        }
        if (relatedOrgs.length > 0 && relatedOrgs.length <= 3) {
          lines.push(`${person} → ${relatedOrgs.join(', ')}`);
        }
      }
    }

    // If no relationships found, just list entity groups
    if (lines.length === 0) {
      if (people && people.size > 0 && people.size <= 20) {
        lines.push(`People: ${[...people].join(', ')}`);
      }
      if (orgs && orgs.size > 0 && orgs.size <= 15) {
        lines.push(`Organizations: ${[...orgs].join(', ')}`);
      }
      if (tools && tools.size > 0 && tools.size <= 15) {
        lines.push(`Tools: ${[...tools].join(', ')}`);
      }
    }

    if (lines.length === 0) return '';

    // Truncate to avoid blowing context budget
    const result = lines.slice(0, 20).join('\n');
    return truncTk(result, 200);
  } catch {
    return '';
  }
}

// --- Helpers

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

async function decomposeQuery(query: string): Promise<string[]> {
  const result = await quickCall(
    `The user wants a complete list. Generate 3 different search queries that would find all instances from different parts of a document. Each query should use different keywords.\n\nOriginal: ${query}\n\nOutput 3 queries, one per line:`,
    80
  );
  const variants = result.split('\n').map(l => l.replace(/^\d+[\.\)]\s*/, '').trim()).filter(l => l.length > 3);
  return variants.length > 0 ? [query, ...variants.slice(0, 3)] : [query];
}

// --- Context Compression

// --- Context Sufficiency Gate
// Insufficient context worsens hallucination (ICLR 2025). Force abstention below threshold.

type ContextSufficiency = 'SUFFICIENT' | 'LOW' | 'INSUFFICIENT' | 'NO_CONTEXT';

function checkContextSufficiency(chunks: string[], avgScore: number): ContextSufficiency {
  if (chunks.length === 0) return 'NO_CONTEXT';
  if (avgScore < 0.25) return 'INSUFFICIENT';
  if (avgScore < 0.45) return 'LOW';
  return 'SUFFICIENT';
}

// --- Negation Handling (code-based, no model)
// 7B models fail at negation; detect it and compute complement in code.

const NEGATION_RE = /\b(not|n'?t|never|no|none|neither|nor|except|other than|besides|excluding|without)\b.*\b(mention|include|list|appear|use|contain|have|reference)\b/i;

function hasNegation(query: string): boolean {
  return NEGATION_RE.test(query);
}

// --- Numerical Query Detection (code computes, not model)

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

  for (let i = 0; i < chunks.length && i < 8; i++) {
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

    // Light cleanup only — do NOT strip sentences or discourse markers.
    // Stripping sentences causes the model to miss answers that use
    // different phrasing than the query (paraphrase/synonym recall failure).
    chunk = chunk.replace(/\s{2,}/g, ' ').trim();

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

// --- Summary System Prompt (synthesis focused)
const SUMMARY_SYSTEM = `You are a world-class analyst. Produce comprehensive, insightful summaries that explain the meaning and significance of the content.`;

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

// --- Core Pipeline

function getParams(frustrated: boolean, isCorrection: boolean, intent: ChatIntent) {
  const maxTk = INTENT_MAX_TOKENS[intent] || CHAT_MAX_TOKENS;
  if (frustrated) return { temperature: 0.5, repeat_penalty: 1.15, presence_penalty: 0.1, top_p: 0.9, top_k: 50, max_tokens: maxTk };
  if (isCorrection) return { temperature: 0.6, repeat_penalty: 1.15, presence_penalty: 0.1, top_p: 0.9, top_k: 50, max_tokens: maxTk };
  return { temperature: 0.7, repeat_penalty: 1.1, presence_penalty: 0.0, top_p: 0.9, top_k: 50, max_tokens: maxTk };
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
  holdServer();
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
    const { isCorrection, assertion } = detectCorrection(rawQuery, lastResponse);
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
      context = compressChunks(chunks, TOKEN_BUDGET.context, excludedEntities);
    } else {
      context = truncTk(rawContext, TOKEN_BUDGET.context);
    }

    // Focused context extraction: find query terms in chunks
    const allChunks = isRAG && chunks ? chunks : context ? [context] : [];
    const extraction = extractFocusedContext(rawQuery, allChunks);

    // Use focused context if term matches found, otherwise full context
    let effectiveContext = extraction.matchCount > 0
      ? extraction.focusedContext
      : context;

    // Entity-enriched context: relationship graph header before raw text
    const entityHeader = buildEntityContext(rawQuery);
    if (entityHeader) {
      effectiveContext = `### Entity Relationships\n${entityHeader}\n\n### Note Content\n${effectiveContext}`;
    }

    // Step 6: Prompt assembly
    const flags = { frustrated, isCorrection, intent };
    const messages = buildMessages(effectiveContext, summary, recent, rewritten, flags);

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
    releaseServer();
  }
}

// --- Streaming Pipeline

async function* streamPipeline(
  rawContext: string,
  rawQuery: string,
  history: ChatMessage[],
  isRAG: boolean,
  chunks?: string[],
): AsyncGenerator<string> {
  await ensureLlamaRunning();
  await acquireInferenceLock();
  holdServer();
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
    const { isCorrection, assertion } = detectCorrection(rawQuery, lastResponse);
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

    // Focused context extraction — narrows context for model, never bypasses it
    const allChunksS = isRAG && chunks ? chunks : context ? [context] : [];
    const extractionS = extractFocusedContext(rawQuery, allChunksS);

    if (extractionS.matchCount > 0) {
      context = extractionS.focusedContext;
    }

    // Entity-enriched context (same as sync pipeline)
    const entityHeaderS = buildEntityContext(rawQuery);
    if (entityHeaderS) {
      context = `### Entity Relationships\n${entityHeaderS}\n\n### Note Content\n${context}`;
    }

    const flags = { frustrated, isCorrection, intent };
    const messages = buildMessages(context, summary, recent, rewritten, flags);

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
    releaseServer();
  }
}

// --- MSR-RAG Chat Pipeline (Multi-Stage Reasoning Retrieval)
// Query Understanding -> Retrieval -> Reasoning -> Targeted Retrieval -> Assembly -> Synthesis -> Verification

import { contextualRetrieveForChat, semanticSearch } from '@/inference/pipeline';

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
  questionType: QuestionType,
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

