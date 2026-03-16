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

// ─── Production Chat System ──────────────────────────────────────────────────
// Architecture:
//   1. Proper multi-turn message format (not flat string concatenation)
//   2. History windowed to last 6 messages (3 turns)
//   3. Older messages auto-summarized to preserve context
//   4. Repetition/presence penalties prevent loops
//   5. Output validation detects and retries bad responses
//   6. Retrieved context injected as system context, not in user message

const MAX_RECENT_MESSAGES = 6; // 3 user + 3 assistant turns
const CHAT_MAX_TOKENS = 250;
const CHAT_TEMPERATURE = 0.6;
const REPEAT_PENALTY = 1.15;
const PRESENCE_PENALTY = 0.15;

const CHAT_SYSTEM_PROMPT = `You are a knowledgeable assistant helping the user understand and discuss a document. Answer accurately based on the provided context. If the answer is not in the context, say so clearly. Be concise and conversational. Never repeat your previous answers.`;

const RAG_SYSTEM_PROMPT = `You are a document analysis assistant.

STRICT RULES:
- Use ONLY the provided context passages to answer.
- If the answer is not in the context, say: "The answer is not available in the provided document."
- Never use outside knowledge.
- Keep responses under 150 words.
- Reference which passage your answer draws from.
- Never repeat a previous answer. Each response must be unique.`;

type ChatMessage = { role: string; content: string };

/** Compress older conversation turns into a brief summary. */
function summarizeOlderMessages(older: ChatMessage[]): string {
  if (older.length === 0) return '';
  const parts: string[] = [];
  for (let i = 0; i < older.length; i += 2) {
    const userMsg = older[i];
    const assistantMsg = older[i + 1];
    if (userMsg) {
      const topic = userMsg.content.slice(0, 100).replace(/\n/g, ' ');
      const answer = assistantMsg
        ? assistantMsg.content.slice(0, 80).replace(/\n/g, ' ')
        : '';
      parts.push(`User asked: "${topic}${userMsg.content.length > 100 ? '...' : ''}"${answer ? ` → Assistant: "${answer}..."` : ''}`);
    }
  }
  return parts.join('\n');
}

/** Window history: keep recent turns, summarize older ones. */
function windowHistory(history: ChatMessage[]): { summary: string; recent: ChatMessage[] } {
  if (history.length <= MAX_RECENT_MESSAGES) {
    return { summary: '', recent: history };
  }
  const older = history.slice(0, -MAX_RECENT_MESSAGES);
  const recent = history.slice(-MAX_RECENT_MESSAGES);
  return { summary: summarizeOlderMessages(older), recent };
}

/** Build proper multi-turn message array for llama-server /v1/chat/completions */
function buildChatMessages(
  systemPrompt: string,
  context: string,
  summary: string,
  recentHistory: ChatMessage[],
  userMessage: string,
): Array<{ role: string; content: string }> {
  // System prompt with context embedded
  let system = systemPrompt;
  if (context) {
    system += `\n\nDocument context:\n"""\n${context}\n"""`;
  }
  if (summary) {
    system += `\n\nConversation summary (earlier messages):\n${summary}`;
  }

  const messages: Array<{ role: string; content: string }> = [
    { role: 'system', content: system },
  ];

  // Add recent history as proper alternating role messages
  for (const msg of recentHistory) {
    messages.push({
      role: msg.role === 'user' ? 'user' : 'assistant',
      content: msg.content,
    });
  }

  // Current user message
  messages.push({ role: 'user', content: userMessage });

  return messages;
}

/** Detect bad output: repetition, empty, or loops */
function validateOutput(text: string): boolean {
  if (!text || text.trim().length < 2) return false;

  // Detect sentence-level repetition (same sentence 3+ times)
  const sentences = text.split(/[.!?]+/).map(s => s.trim().toLowerCase()).filter(s => s.length > 10);
  const seen = new Map<string, number>();
  for (const s of sentences) {
    seen.set(s, (seen.get(s) || 0) + 1);
    if ((seen.get(s) || 0) >= 3) return false;
  }

  // Detect phrase-level loops (20+ char substring repeated 3+ times)
  const lower = text.toLowerCase();
  for (let len = 20; len <= 60; len += 10) {
    for (let i = 0; i <= lower.length - len; i += 10) {
      const phrase = lower.slice(i, i + len);
      let count = 0;
      let pos = 0;
      while ((pos = lower.indexOf(phrase, pos)) !== -1) { count++; pos += 1; }
      if (count >= 3) return false;
    }
  }

  return true;
}

/** Core chat call with proper format, penalties, and validation */
async function callChat(
  systemPrompt: string,
  context: string,
  userMessage: string,
  history: ChatMessage[],
  maxTokens: number = CHAT_MAX_TOKENS,
): Promise<string> {
  await ensureLlamaRunning();
  await acquireInferenceLock();
  resetIdleTimer();
  const startTime = Date.now();

  try {
    const { summary, recent } = windowHistory(history);
    const messages = buildChatMessages(systemPrompt, context, summary, recent, userMessage);

    const body = {
      messages,
      temperature: CHAT_TEMPERATURE,
      repeat_penalty: REPEAT_PENALTY,
      presence_penalty: PRESENCE_PENALTY,
      top_p: 0.9,
      max_tokens: maxTokens,
      stream: false,
    };

    const res = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`LLM error: ${res.status}`);

    const data = await res.json();
    let response = data.choices?.[0]?.message?.content || '';

    // Validate output — retry once if bad
    if (!validateOutput(response)) {
      console.warn('[Chat] Bad output detected (repetition/empty), retrying...');
      const retryBody = { ...body, temperature: 0.8, repeat_penalty: 1.3 };
      const retryRes = await fetch(`${LLAMA_URL}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(retryBody),
      });
      if (retryRes.ok) {
        const retryData = await retryRes.json();
        response = retryData.choices?.[0]?.message?.content || response;
      }
    }

    console.log(`[Chat] ${Date.now() - startTime}ms | turns=${history.length / 2} | summary=${summary ? 'yes' : 'no'}`);
    return response;
  } finally {
    recordInferenceTime(Date.now() - startTime);
    releaseInferenceLock();
  }
}

/** Core streaming chat with anti-loop protection */
async function* streamChat(
  systemPrompt: string,
  context: string,
  userMessage: string,
  history: ChatMessage[],
  maxTokens: number = CHAT_MAX_TOKENS,
): AsyncGenerator<string> {
  await ensureLlamaRunning();
  await acquireInferenceLock();
  resetIdleTimer();
  const startTime = Date.now();

  try {
    const { summary, recent } = windowHistory(history);
    const messages = buildChatMessages(systemPrompt, context, summary, recent, userMessage);

    const body = {
      messages,
      temperature: CHAT_TEMPERATURE,
      repeat_penalty: REPEAT_PENALTY,
      presence_penalty: PRESENCE_PENALTY,
      top_p: 0.9,
      max_tokens: maxTokens,
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
    let lastYielded = '';

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

            // Anti-loop: detect if last 60 chars are repeating
            if (accumulated.length > 120) {
              const tail = accumulated.slice(-60);
              const before = accumulated.slice(-120, -60);
              if (tail === before) {
                console.warn('[Chat] Loop detected in stream, stopping');
                return;
              }
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

// ─── Public Chat API (drop-in replacements) ──────────────────────────────────

export async function chatWithContext(
  documentText: string,
  userMessage: string,
  history: ChatMessage[],
): Promise<string> {
  const context = cleanInputText(documentText);
  return callChat(CHAT_SYSTEM_PROMPT, context, userMessage, history);
}

export async function* chatWithContextStream(
  documentText: string,
  userMessage: string,
  history: ChatMessage[],
): AsyncGenerator<string> {
  const context = cleanInputText(documentText);
  yield* streamChat(CHAT_SYSTEM_PROMPT, context, userMessage, history);
}

export async function chatWithRAG(
  chunks: string[],
  userMessage: string,
  history: ChatMessage[],
): Promise<string> {
  const context = chunks.map((c, i) => `[Passage ${i + 1}]\n${c}`).join('\n\n');
  return callChat(RAG_SYSTEM_PROMPT, context, userMessage, history);
}

export async function* chatWithRAGStream(
  chunks: string[],
  userMessage: string,
  history: ChatMessage[],
): AsyncGenerator<string> {
  const context = chunks.map((c, i) => `[Passage ${i + 1}]\n${c}`).join('\n\n');
  yield* streamChat(RAG_SYSTEM_PROMPT, context, userMessage, history);
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
