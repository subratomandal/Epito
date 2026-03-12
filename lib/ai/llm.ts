const LLAMA_PORT = process.env.LLAMA_SERVER_PORT || '8080';
const LLAMA_URL = `http://127.0.0.1:${LLAMA_PORT}`;
const MODEL = 'mistral-7b-instruct';

console.log(`[LLM] Configured: url=${LLAMA_URL}, model=${MODEL}`);

let inferenceActive = false;
const inferenceQueue: Array<{ resolve: () => void }> = [];

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
  const cooldown = computeThermalCooldown();
  if (cooldown > 0) await sleep(cooldown);

  const effectiveMax = adaptiveMaxTokens(maxTokens);

  await acquireInferenceLock();
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
    return data.choices[0].message.content;
  } finally {
    recordInferenceTime(Date.now() - startTime);
    releaseInferenceLock();
  }
}

async function* callLlamaStream(prompt: string, systemPrompt: string, maxTokens?: number): AsyncGenerator<string> {
  const cooldown = computeThermalCooldown();
  if (cooldown > 0) await sleep(cooldown);

  const effectiveMax = adaptiveMaxTokens(maxTokens);

  await acquireInferenceLock();
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

const CHAT_SYSTEM_PROMPT = `You are a knowledgeable assistant helping the user understand and discuss a document. The document content is provided as context. Answer questions thoroughly and accurately based on the document. If the answer is not in the document, say so clearly. Be conversational but substantive.`;

function buildChatPrompt(
  documentText: string,
  userMessage: string,
  history: { role: string; content: string }[],
): string {
  let prompt = `Document context:\n"""\n${documentText}\n"""\n\n`;

  if (history.length > 0) {
    prompt += 'Conversation so far:\n';
    for (const msg of history) {
      prompt += `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}\n`;
    }
    prompt += '\n';
  }

  prompt += `User: ${userMessage}\n\nProvide a thorough, helpful response:`;
  return prompt;
}

export async function chatWithContext(
  documentText: string,
  userMessage: string,
  history: { role: string; content: string }[],
): Promise<string> {
  const cleaned = cleanInputText(documentText);
  const prompt = buildChatPrompt(cleaned, userMessage, history);
  return callLLM(prompt, CHAT_SYSTEM_PROMPT, MAX_OUTPUT_TOKENS);
}

export async function* chatWithContextStream(
  documentText: string,
  userMessage: string,
  history: { role: string; content: string }[],
): AsyncGenerator<string> {
  const cleaned = cleanInputText(documentText);
  const prompt = buildChatPrompt(cleaned, userMessage, history);
  yield* streamLLM(prompt, CHAT_SYSTEM_PROMPT, MAX_OUTPUT_TOKENS);
}

const RAG_SYSTEM_PROMPT = `You are a document analysis assistant.

STRICT RULES:
- Use ONLY the provided context passages to answer the question.
- If the answer is not contained in the context, respond with: "The answer is not available in the provided document."
- Never use outside knowledge or make assumptions beyond what the context states.
- Keep your response under 150 words. Be concise and precise.
- Reference which passage(s) your answer draws from (e.g., "According to Passage 1...").
- Be conversational but factual.`;

function buildRAGPrompt(
  chunks: string[],
  userMessage: string,
  history: { role: string; content: string }[],
): string {
  const context = chunks.map((c, i) => `[Passage ${i + 1}]\n${c}`).join('\n\n');

  let prompt = `Context:\n"""\n${context}\n"""\n\n`;

  if (history.length > 0) {
    prompt += 'Conversation so far:\n';
    for (const msg of history) {
      prompt += `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}\n`;
    }
    prompt += '\n';
  }

  prompt += `User Question:\n${userMessage}\n\nInstruction:\nAnswer the question strictly using the context above. Do not add external information. If the context does not contain the answer, say that the answer is not available in the document.`;
  return prompt;
}

export async function chatWithRAG(
  chunks: string[],
  userMessage: string,
  history: { role: string; content: string }[],
): Promise<string> {
  const prompt = buildRAGPrompt(chunks, userMessage, history);
  return callLLM(prompt, RAG_SYSTEM_PROMPT, MAX_OUTPUT_TOKENS);
}

export async function* chatWithRAGStream(
  chunks: string[],
  userMessage: string,
  history: { role: string; content: string }[],
): AsyncGenerator<string> {
  const prompt = buildRAGPrompt(chunks, userMessage, history);
  yield* streamLLM(prompt, RAG_SYSTEM_PROMPT, MAX_OUTPUT_TOKENS);
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
