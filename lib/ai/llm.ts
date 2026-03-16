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

const SYSTEM_PROMPT = `You are a world-class analyst and educator. You produce comprehensive, deeply insightful, publication-quality outputs. You never rush, never abbreviate, and never produce shallow work. Every response should demonstrate expert-level understanding and provide genuine value that goes beyond what the user could derive on their own.`;

const CHUNK_INSIGHT_PROMPT = (sections: string) =>
`Analyze each text section below with expert-level depth. For each section, extract:

1. The core argument, thesis, or main idea being communicated
2. Key facts, evidence, data points, or supporting claims
3. Non-obvious implications — what does this mean in a broader context?
4. Connections between ideas across different sections
5. Any assumptions, limitations, or counterarguments implied

Do NOT copy or paraphrase the original text. Extract the underlying meaning and significance. Be thorough and specific — vague summaries are useless.

Text sections:
"""
${sections}
"""`;

const SYNTHESIS_PROMPT = (insights: string) =>
`You are synthesizing extracted insights into a comprehensive, well-organized analysis.

Create a structured summary following this format:
- Start with a single overview sentence that captures the central thesis or theme
- Group related insights into logical categories
- Use "-" for main points
- Use "  -" (two spaces + dash) for supporting details under each main point
- Every main point should contain at least one supporting detail
- Cover ALL significant ideas — do not drop or merge important distinct points
- Highlight the most important and non-obvious findings
- Remove genuine redundancy but preserve nuance and distinct perspectives
- Write in clear, analytical language — rephrase for clarity, never copy original text

Do NOT use markdown formatting (no bold, italic, headers, numbering).
Do NOT include meta-commentary like "Here is the summary" or "In conclusion."
Output the structured analysis directly.

Extracted insights:
"""
${insights}
"""`;

export async function summarizeChunks(chunks: string[]): Promise<{ summary: string; keyPoints: string[] } | null> {
  try {
    const sections = chunks.map((c, i) => `[Section ${i + 1}]\n${c}`).join('\n\n');

    const insights = await callLLM(CHUNK_INSIGHT_PROMPT(sections), SYSTEM_PROMPT);

    const finalResponse = await callLLM(SYNTHESIS_PROMPT(insights.trim()), SYSTEM_PROMPT);

    return { summary: cleanSummaryOutput(finalResponse), keyPoints: [] };
  } catch (err) {
    console.error('[LLM] Chunk summarize error:', err);
    throw err;
  }
}

export async function* summarizeChunksStream(chunks: string[]): AsyncGenerator<string> {
  const sections = chunks.map((c, i) => `[Section ${i + 1}]\n${c}`).join('\n\n');

  const insights = await callLLM(CHUNK_INSIGHT_PROMPT(sections), SYSTEM_PROMPT);

  for await (const text of streamLLM(SYNTHESIS_PROMPT(insights.trim()), SYSTEM_PROMPT)) {
    yield cleanSummaryOutput(text);
  }
}

const SUMMARIZE_PROMPT = (text: string) =>
`Analyze the following text and produce a comprehensive structured analysis.

Create a structured summary following this format:
- Start with a single overview sentence that captures the central thesis or theme
- Group related ideas into logical categories
- Use "-" for main points
- Use "  -" (two spaces + dash) for supporting details under each main point
- Every main point should contain at least one supporting detail
- Extract deep insights — what does this text really mean? What are the implications?
- Do not just rephrase sentences — analyze, synthesize, and provide genuine understanding
- Cover ALL significant ideas comprehensively

Do NOT use markdown formatting (no bold, italic, headers, numbering).
Do NOT include meta-commentary like "Here is the summary."
Output the structured analysis directly.

Text:
"""
${text}
"""`;

export async function summarizeText(text: string): Promise<{ summary: string; keyPoints: string[] } | null> {
  try {
    const cleaned = cleanInputText(text);
    const response = await callLLM(SUMMARIZE_PROMPT(cleaned), SYSTEM_PROMPT);
    return { summary: cleanSummaryOutput(response), keyPoints: [] };
  } catch (err) {
    console.error('[LLM] Summarize error:', err);
    throw err;
  }
}

export async function* summarizeTextStream(text: string): AsyncGenerator<string> {
  const cleaned = cleanInputText(text);
  for await (const chunk of streamLLM(SUMMARIZE_PROMPT(cleaned), SYSTEM_PROMPT)) {
    yield cleanSummaryOutput(chunk);
  }
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
// Mistral 7B Chat System — Tuned for 4096-token sliding window attention
//
// Architecture:
//   - Strict token budget enforcement (4096 total, 250 response cap)
//   - History capped at 4 messages (2 turns) — Mistral can't leverage more
//   - Older turns summarized via one-shot Mistral call
//   - Query rewriting resolves coreferences ("what about that?" → standalone)
//   - Dual-anchor grounding: instructions at TOP + reminder at BOTTOM
//   - Aggressive anti-loop: 8-token ngram detection in streaming
//   - Output validation with single retry on failure
//   - KV cache rebuild every 6 turns
// ═══════════════════════════════════════════════════════════════════════════════

type ChatMessage = { role: string; content: string };

// ─── Token Budget (4096 total) ───────────────────────────────────────────────
const TOKEN_BUDGET = {
  system: 200,
  summary: 150,
  context: 800,
  recentChat: 600,
  query: 150,
  // response headroom: ~2196, capped at 250
};
const CHAT_MAX_TOKENS = 250;
const MAX_RECENT_MESSAGES = 4; // 2 user + 2 assistant (Mistral can't use more)
const KV_CACHE_RESET_INTERVAL = 6; // Reset every 6 turns

// ─── Decoding Parameters (Mistral 7B specific) ──────────────────────────────
const CHAT_PARAMS = {
  temperature: 0.25,       // Low for factual grounding
  repeat_penalty: 1.2,     // Critical — Mistral loops without this
  presence_penalty: 0.15,
  top_p: 0.9,
  top_k: 40,
  max_tokens: CHAT_MAX_TOKENS,
};

// ─── System Prompt (<200 tokens, short imperative sentences) ─────────────────
const CHAT_SYSTEM = `You are Epito, a note assistant. Answer questions using only the provided Notes section. Be concise and factual. If the Notes do not contain the answer, say "I don't have that information in your notes." Do not invent facts. Do not repeat yourself.`;

const RAG_SYSTEM = `You are Epito, a document assistant. Answer using ONLY the provided passages. Reference which passage you use. If no passage answers the question, say "I don't have that information in your notes." Be concise. Do not invent facts. Do not repeat yourself.`;

// ─── Grounding Anchor (placed at END of prompt, re-anchors instructions) ─────
const GROUNDING_ANCHOR = `Respond using only the information in Notes and Recent Chat. If the answer is not there, say you don't have that information.`;

// ─── Token Estimation ────────────────────────────────────────────────────────

function countTokens(text: string): number {
  // Mistral tokenizer averages ~1.3 tokens per word
  return Math.ceil(text.split(/\s+/).length * 1.3);
}

function truncateToTokens(text: string, maxTokens: number): string {
  const words = text.split(/\s+/);
  const maxWords = Math.floor(maxTokens / 1.3);
  if (words.length <= maxWords) return text;
  return words.slice(0, maxWords).join(' ') + '...';
}

// ─── Conversation Summary (Mistral one-shot call) ────────────────────────────

let cachedSummary = '';
let cachedSummaryTurnCount = 0;

async function summarizeHistory(older: ChatMessage[]): Promise<string> {
  if (older.length === 0) return '';

  // Build conversation text for summarization
  const convText = older.map(m =>
    `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content.slice(0, 150)}`
  ).join('\n');

  try {
    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [
          { role: 'user', content: `Summarize this conversation in 2-3 sentences. Keep: user goals, key facts, topics discussed.\n\n${convText}` },
        ],
        temperature: 0.1,
        max_tokens: 100,
        repeat_penalty: 1.1,
        stream: false,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      const summary = data.choices?.[0]?.message?.content || '';
      return truncateToTokens(summary, TOKEN_BUDGET.summary);
    }
  } catch (e) {
    console.warn('[Chat] Summary call failed, using local compression:', e);
  }

  // Fallback: local compression (no model call)
  return older
    .filter(m => m.role === 'user')
    .map(m => m.content.slice(0, 60).replace(/\n/g, ' '))
    .join('. ')
    .slice(0, 200);
}

// ─── Query Rewriting (resolve coreferences for Mistral) ──────────────────────

async function rewriteQuery(rawQuery: string, recentMessages: ChatMessage[]): Promise<string> {
  // Skip rewriting for simple/short queries
  if (rawQuery.split(/\s+/).length > 15 || recentMessages.length === 0) {
    return rawQuery;
  }

  const recent = recentMessages.slice(-4).map(m =>
    `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content.slice(0, 100)}`
  ).join('\n');

  try {
    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [
          { role: 'user', content: `Rewrite this question as a clear standalone question using the conversation context.\n\nContext:\n${recent}\n\nQuestion: ${rawQuery}\n\nOutput only the rewritten question, nothing else.` },
        ],
        temperature: 0.1,
        max_tokens: 60,
        repeat_penalty: 1.1,
        stream: false,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      const rewritten = (data.choices?.[0]?.message?.content || '').trim();
      if (rewritten && rewritten.length > 5 && rewritten.length < 300) {
        console.log(`[Chat] Query rewrite: "${rawQuery}" → "${rewritten}"`);
        return rewritten;
      }
    }
  } catch {}

  return rawQuery;
}

// ─── Context Compression ─────────────────────────────────────────────────────

function compressContext(chunks: string[], maxTokens: number): string {
  if (chunks.length === 0) return '';

  const compressed: string[] = [];
  let totalTokens = 0;

  for (let i = 0; i < chunks.length && i < 5; i++) {
    let chunk = chunks[i];

    // Strip filler/hedging
    chunk = chunk
      .replace(/\b(however|moreover|furthermore|additionally|in addition|it is worth noting that|it should be noted that)\b/gi, '')
      .replace(/\s{2,}/g, ' ')
      .trim();

    // Truncate individual chunks to ~3 sentences
    const sentences = chunk.split(/(?<=[.!?])\s+/);
    if (sentences.length > 3) {
      chunk = sentences.slice(0, 3).join(' ');
    }

    const tokens = countTokens(chunk);
    if (totalTokens + tokens > maxTokens) {
      // Try to fit partial
      const remaining = maxTokens - totalTokens;
      if (remaining > 30) {
        compressed.push(`[${i + 1}] ${truncateToTokens(chunk, remaining)}`);
      }
      break;
    }

    compressed.push(`[${i + 1}] ${chunk}`);
    totalTokens += tokens;
  }

  return compressed.join('\n\n');
}

// ─── Window History + Summary ────────────────────────────────────────────────

async function prepareHistory(history: ChatMessage[], turnCount: number): Promise<{
  summary: string;
  recent: ChatMessage[];
  needsCacheReset: boolean;
}> {
  const needsCacheReset = turnCount > 0 && turnCount % KV_CACHE_RESET_INTERVAL === 0;

  if (history.length <= MAX_RECENT_MESSAGES) {
    return { summary: cachedSummary, recent: history, needsCacheReset };
  }

  const older = history.slice(0, -MAX_RECENT_MESSAGES);
  const recent = history.slice(-MAX_RECENT_MESSAGES);

  // Only re-summarize if history grew since last summary
  if (older.length > cachedSummaryTurnCount) {
    cachedSummary = await summarizeHistory(older);
    cachedSummaryTurnCount = older.length;
    console.log(`[Chat] Summarized ${older.length} older messages`);
  }

  // Truncate recent messages to fit token budget
  const truncatedRecent = recent.map(m => ({
    ...m,
    content: truncateToTokens(m.content, TOKEN_BUDGET.recentChat / MAX_RECENT_MESSAGES),
  }));

  return { summary: cachedSummary, recent: truncatedRecent, needsCacheReset };
}

// ─── Prompt Assembly (Mistral [INST] format via messages API) ────────────────
// llama-server auto-applies Mistral's chat template when using /v1/chat/completions

function buildPromptMessages(
  systemPrompt: string,
  summary: string,
  context: string,
  recentHistory: ChatMessage[],
  rewrittenQuery: string,
): Array<{ role: string; content: string }> {
  // Build system content with dual-anchor grounding
  let system = truncateToTokens(systemPrompt, TOKEN_BUDGET.system);

  if (summary) {
    system += `\n\n### Summary\n${summary}`;
  }
  if (context) {
    system += `\n\n### Notes\n${context}`;
  }

  const messages: Array<{ role: string; content: string }> = [
    { role: 'system', content: system },
  ];

  // Recent chat as proper alternating messages
  for (const msg of recentHistory) {
    messages.push({
      role: msg.role === 'user' ? 'user' : 'assistant',
      content: msg.content,
    });
  }

  // User query with grounding anchor at the bottom (dual-anchor pattern)
  messages.push({
    role: 'user',
    content: `${rewrittenQuery}\n\n${GROUNDING_ANCHOR}`,
  });

  return messages;
}

// ─── Anti-Loop Detection (8-token ngram, critical for Mistral 7B) ────────────

function detectLoop(text: string): boolean {
  if (text.length < 80) return false;

  // Check for 8+ word sequences appearing 3+ times
  const words = text.toLowerCase().split(/\s+/);
  if (words.length < 24) return false;

  const ngrams = new Map<string, number>();
  for (let i = 0; i <= words.length - 8; i++) {
    const gram = words.slice(i, i + 8).join(' ');
    ngrams.set(gram, (ngrams.get(gram) || 0) + 1);
    if ((ngrams.get(gram) || 0) >= 3) return true;
  }
  return false;
}

// ─── Output Validation ───────────────────────────────────────────────────────

function validateChatOutput(text: string, query: string, context: string): 'ok' | 'empty' | 'repetition' | 'off-topic' {
  if (!text || text.trim().length < 5) return 'empty';
  if (detectLoop(text)) return 'repetition';

  // Check sentence-level repeats (same sentence 2+ times)
  const sentences = text.split(/[.!?]+/).map(s => s.trim().toLowerCase()).filter(s => s.length > 15);
  const seen = new Set<string>();
  for (const s of sentences) {
    if (seen.has(s)) return 'repetition';
    seen.add(s);
  }

  // Off-topic check: response shares no significant words with query+context
  if (query && context) {
    const queryWords = new Set(query.toLowerCase().split(/\s+/).filter(w => w.length > 4));
    const contextWords = new Set(context.toLowerCase().split(/\s+/).filter(w => w.length > 4).slice(0, 100));
    const responseWords = text.toLowerCase().split(/\s+/).filter(w => w.length > 4);
    const overlap = responseWords.filter(w => queryWords.has(w) || contextWords.has(w));
    if (overlap.length === 0 && responseWords.length > 10) return 'off-topic';
  }

  return 'ok';
}

const FALLBACK_RESPONSE = "I couldn't find a clear answer in your notes. Try rephrasing your question.";

// ─── Core Chat Call ──────────────────────────────────────────────────────────

let chatTurnCounter = 0;

async function callChat(
  systemPrompt: string,
  rawContext: string,
  rawQuery: string,
  history: ChatMessage[],
): Promise<string> {
  await ensureLlamaRunning();
  await acquireInferenceLock();
  resetIdleTimer();
  const startTime = Date.now();

  try {
    chatTurnCounter++;
    const { summary, recent, needsCacheReset } = await prepareHistory(history, chatTurnCounter);

    if (needsCacheReset) {
      console.log(`[Chat] KV cache reset at turn ${chatTurnCounter}`);
      // Force fresh prompt rebuild — no stale KV cache
      try {
        await fetch(`${LLAMA_URL}/slots/0?action=erase`, { method: 'POST', signal: AbortSignal.timeout(2000) });
      } catch {}
    }

    // Rewrite query to resolve coreferences
    const rewrittenQuery = await rewriteQuery(rawQuery, recent);

    // Compress context to fit token budget
    const context = truncateToTokens(rawContext, TOKEN_BUDGET.context);

    const messages = buildPromptMessages(systemPrompt, summary, context, recent, rewrittenQuery);

    const body = {
      messages,
      ...CHAT_PARAMS,
      stream: false,
    };

    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`LLM error: ${res.status}`);

    const data = await res.json();
    let response = (data.choices?.[0]?.message?.content || '').trim();

    // Validate output
    const validation = validateChatOutput(response, rewrittenQuery, context);
    if (validation !== 'ok') {
      console.warn(`[Chat] Validation failed: ${validation}. Retrying...`);
      const retryBody = {
        ...body,
        temperature: Math.min(CHAT_PARAMS.temperature + 0.15, 0.5),
        repeat_penalty: 1.25,
      };
      const retryRes = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(retryBody),
      });
      if (retryRes.ok) {
        const retryData = await retryRes.json();
        const retryResponse = (retryData.choices?.[0]?.message?.content || '').trim();
        if (validateChatOutput(retryResponse, rewrittenQuery, context) === 'ok') {
          response = retryResponse;
        } else {
          response = FALLBACK_RESPONSE;
        }
      } else {
        response = FALLBACK_RESPONSE;
      }
    }

    console.log(`[Chat] ${Date.now() - startTime}ms | turn=${chatTurnCounter} | summary=${summary ? 'yes' : 'no'} | rewrite=${rewrittenQuery !== rawQuery}`);
    return response;
  } finally {
    recordInferenceTime(Date.now() - startTime);
    releaseInferenceLock();
  }
}

// ─── Core Streaming Chat ─────────────────────────────────────────────────────

async function* streamChat(
  systemPrompt: string,
  rawContext: string,
  rawQuery: string,
  history: ChatMessage[],
): AsyncGenerator<string> {
  await ensureLlamaRunning();
  await acquireInferenceLock();
  resetIdleTimer();
  const startTime = Date.now();

  try {
    chatTurnCounter++;
    const { summary, recent, needsCacheReset } = await prepareHistory(history, chatTurnCounter);

    if (needsCacheReset) {
      try {
        await fetch(`${LLAMA_URL}/slots/0?action=erase`, { method: 'POST', signal: AbortSignal.timeout(2000) });
      } catch {}
    }

    const rewrittenQuery = await rewriteQuery(rawQuery, recent);
    const context = truncateToTokens(rawContext, TOKEN_BUDGET.context);
    const messages = buildPromptMessages(systemPrompt, summary, context, recent, rewrittenQuery);

    const body = {
      messages,
      ...CHAT_PARAMS,
      stream: true,
    };

    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`LLM error: ${res.status}`);

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
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const payload = trimmed.slice(6);
        if (payload === '[DONE]') continue;
        try {
          const data = JSON.parse(payload);
          const content = data.choices?.[0]?.delta?.content;
          if (content) {
            accumulated += content;

            // Anti-loop: 8-word ngram detection (critical for Mistral 7B)
            if (detectLoop(accumulated)) {
              console.warn('[Chat] Loop detected in stream at', accumulated.length, 'chars');
              // Truncate at the clean boundary before the loop started
              const words = accumulated.split(/\s+/);
              const cleanEnd = Math.max(words.length - 16, Math.floor(words.length * 0.7));
              accumulated = words.slice(0, cleanEnd).join(' ');
              yield accumulated;
              return;
            }

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

// ─── Public Chat API ─────────────────────────────────────────────────────────

export async function chatWithContext(
  documentText: string,
  userMessage: string,
  history: ChatMessage[],
): Promise<string> {
  const context = cleanInputText(documentText);
  return callChat(CHAT_SYSTEM, context, userMessage, history);
}

export async function* chatWithContextStream(
  documentText: string,
  userMessage: string,
  history: ChatMessage[],
): AsyncGenerator<string> {
  const context = cleanInputText(documentText);
  yield* streamChat(CHAT_SYSTEM, context, userMessage, history);
}

export async function chatWithRAG(
  chunks: string[],
  userMessage: string,
  history: ChatMessage[],
): Promise<string> {
  const context = compressContext(chunks, TOKEN_BUDGET.context);
  return callChat(RAG_SYSTEM, context, userMessage, history);
}

export async function* chatWithRAGStream(
  chunks: string[],
  userMessage: string,
  history: ChatMessage[],
): AsyncGenerator<string> {
  const context = compressContext(chunks, TOKEN_BUDGET.context);
  yield* streamChat(RAG_SYSTEM, context, userMessage, history);
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
