import { randomUUID } from 'crypto';
import * as db from '@/lib/database';
import { generateEmbedding, initEmbeddings, EMBEDDING_DIM } from './embeddings';
import { VectorIndex, cosineSimilarity } from './vector';
import { cleanInputText } from './llm';
import type { Note, SearchResult, RelatedNote, Topic, AISummary, SourceType, ContextualMatch, ChatRetrievalResult } from '@/lib/types';
import { stripHtml } from '@/lib/utils';

const chunkIndex = new VectorIndex(EMBEDDING_DIM);
const noteIndex = new VectorIndex(EMBEDDING_DIM);
let initialized = false;

export async function initPipeline(): Promise<void> {
  if (initialized) return;
  await initEmbeddings();

  const chunkEmbs = db.getAllEmbeddings();
  chunkIndex.addBatch(chunkEmbs.map(e => ({ id: e.chunkId, vector: e.vector })));

  const noteEmbs = db.getAllNoteEmbeddings();
  noteIndex.addBatch(noteEmbs.map(e => ({ id: e.noteId, vector: e.vector })));

  initialized = true;
  console.log(`[AI] Pipeline ready. Chunks: ${chunkIndex.size}, Notes: ${noteIndex.size}`);
}

const CHUNK_MIN_WORDS = 300;
const CHUNK_MAX_WORDS = 500;
const CHUNK_OVERLAP_WORDS = 75;

function countWords(text: string): number {
  return text.split(/\s+/).filter(w => w.length > 0).length;
}

function splitSentences(text: string): string[] {
  const raw = text.split(/(?<=[.!?])\s+|(?:\n\s*\n)/);
  return raw.map(s => s.trim()).filter(s => s.length > 0);
}

function chunkText(text: string): { content: string; startOffset: number; endOffset: number }[] {
  const cleaned = cleanInputText(text);
  const sentences = splitSentences(cleaned);

  if (sentences.length === 0) {
    return cleaned.length > 0
      ? [{ content: cleaned, startOffset: 0, endOffset: cleaned.length }]
      : [];
  }

  const chunks: { content: string; startOffset: number; endOffset: number }[] = [];
  let currentSentences: string[] = [];
  let currentWords = 0;
  let offset = 0;
  let chunkStart = 0;

  for (let si = 0; si < sentences.length; si++) {
    const sentence = sentences[si];
    const sentenceWords = countWords(sentence);

    if (currentWords + sentenceWords > CHUNK_MAX_WORDS && currentWords >= CHUNK_MIN_WORDS) {
      const content = currentSentences.join(' ');
      chunks.push({ content, startOffset: chunkStart, endOffset: offset });

      let overlapWords = 0;
      let overlapStart = currentSentences.length;
      for (let j = currentSentences.length - 1; j >= 0; j--) {
        const w = countWords(currentSentences[j]);
        if (overlapWords + w > CHUNK_OVERLAP_WORDS) break;
        overlapWords += w;
        overlapStart = j;
      }

      const overlapSentences = currentSentences.slice(overlapStart);
      currentSentences = [...overlapSentences];
      currentWords = overlapWords;
      chunkStart = offset - overlapSentences.join(' ').length;
    }

    currentSentences.push(sentence);
    currentWords += sentenceWords;
    offset += sentence.length + 1;
  }

  if (currentSentences.length > 0) {
    const content = currentSentences.join(' ');
    if (currentWords < CHUNK_MIN_WORDS / 2 && chunks.length > 0) {
      const lastChunk = chunks[chunks.length - 1];
      chunks[chunks.length - 1] = {
        content: lastChunk.content + ' ' + content,
        startOffset: lastChunk.startOffset,
        endOffset: offset,
      };
    } else {
      chunks.push({ content, startOffset: chunkStart, endOffset: offset });
    }
  }

  return chunks.length > 0 ? chunks : [{ content: cleaned, startOffset: 0, endOffset: cleaned.length }];
}

export async function processNote(noteId: string): Promise<void> {
  await initPipeline();

  const note = db.getNote(noteId);
  if (!note) return;

  const plainText = note.plain_text || stripHtml(note.content);

  db.deleteChunksBySource(noteId, 'note');

  const chunks = chunkText(plainText);
  const chunkIds: string[] = [];

  for (const chunk of chunks) {
    const id = db.insertChunk(noteId, chunk.content, chunk.startOffset, chunk.endOffset, 'note');
    chunkIds.push(id);
  }
  db.updateNoteChunkCount(noteId, chunkIds.length);

  for (let i = 0; i < chunkIds.length; i++) {
    const emb = await generateEmbedding(chunks[i].content);
    db.insertEmbedding(chunkIds[i], emb);
    chunkIndex.add(chunkIds[i], emb);
  }

  const noteText = `${note.title}. ${plainText.slice(0, 1000)}`;
  const noteEmb = await generateEmbedding(noteText);
  db.insertNoteEmbedding(noteId, noteEmb);
  noteIndex.add(noteId, noteEmb);

  scheduleRebuildTopicsAndLinks();
}

export async function processDocument(docId: string): Promise<void> {
  await initPipeline();

  const doc = db.getDocument(docId);
  if (!doc) return;

  try {
    db.deleteChunksBySource(docId, 'document');

    const text = doc.plain_text;
    if (!text.trim()) {
      db.updateDocument(docId, { status: 'error' });
      return;
    }

    const chunks = chunkText(text);
    const chunkIds: string[] = [];

    for (const chunk of chunks) {
      const id = db.insertChunk(docId, chunk.content, chunk.startOffset, chunk.endOffset, 'document');
      chunkIds.push(id);
    }
    db.updateDocument(docId, { chunk_count: chunkIds.length });

    for (let i = 0; i < chunkIds.length; i++) {
      const emb = await generateEmbedding(chunks[i].content);
      db.insertEmbedding(chunkIds[i], emb);
      chunkIndex.add(chunkIds[i], emb);
    }

    const docText = `${doc.filename}. ${text.slice(0, 1000)}`;
    const docEmb = await generateEmbedding(docText);
    db.insertNoteEmbedding(docId, docEmb);
    noteIndex.add(docId, docEmb);

    db.updateDocument(docId, { status: 'ready' });
    console.log(`[AI] Document processed: ${doc.filename} (${chunkIds.length} chunks)`);

    scheduleRebuildTopicsAndLinks();
  } catch (err) {
    console.error(`[AI] Document processing error:`, err);
    db.updateDocument(docId, { status: 'error' });
  }
}

export async function processImage(imageId: string): Promise<void> {
  await initPipeline();

  const img = db.getImage(imageId);
  if (!img) return;

  try {
    db.deleteChunksBySource(imageId, 'image');

    const text = img.ocr_text;
    if (!text.trim()) {
      db.updateImage(imageId, { status: 'ready', chunk_count: 0 });
      return;
    }

    const chunks = chunkText(text);
    const chunkIds: string[] = [];

    for (const chunk of chunks) {
      const id = db.insertChunk(imageId, chunk.content, chunk.startOffset, chunk.endOffset, 'image');
      chunkIds.push(id);
    }
    db.updateImage(imageId, { chunk_count: chunkIds.length });

    for (let i = 0; i < chunkIds.length; i++) {
      const emb = await generateEmbedding(chunks[i].content);
      db.insertEmbedding(chunkIds[i], emb);
      chunkIndex.add(chunkIds[i], emb);
    }

    const imgText = `${img.filename}. ${text.slice(0, 1000)}`;
    const imgEmb = await generateEmbedding(imgText);
    db.insertNoteEmbedding(imageId, imgEmb);
    noteIndex.add(imageId, imgEmb);

    db.updateImage(imageId, { status: 'ready' });
    console.log(`[AI] Image processed: ${img.filename} (${chunkIds.length} chunks)`);

    scheduleRebuildTopicsAndLinks();
  } catch (err) {
    console.error(`[AI] Image processing error:`, err);
    db.updateImage(imageId, { status: 'error' });
  }
}

export async function semanticSearch(query: string, topK = 20): Promise<SearchResult[]> {
  topK = Math.min(Math.max(1, topK), 100);
  await initPipeline();
  const queryEmb = await generateEmbedding(query);
  const hits = chunkIndex.search(queryEmb, topK * 2, 0.15);

  const allNotes = db.getAllNotes();
  const noteMap = new Map(allNotes.map(n => [n.id, n]));

  const allDocs = db.getAllDocuments();
  const docMap = new Map(allDocs.map(d => [d.id, d]));

  const allImages = db.getAllImages();
  const imageMap = new Map(allImages.map(i => [i.id, i]));

  const allTopics = db.getAllTopics();
  const hitChunks = db.getChunksByIds(hits.map(h => h.id));
  const chunkMap = new Map(hitChunks.map(c => [c.id, c]));

  const results: SearchResult[] = [];
  const seenSources = new Set<string>();

  for (const hit of hits) {
    const chunk = chunkMap.get(hit.id);
    if (!chunk) continue;

    const sourceKey = `${chunk.source_type}:${chunk.note_id}`;
    if (seenSources.has(sourceKey) && results.length >= topK) continue;
    seenSources.add(sourceKey);

    const sourceType = (chunk.source_type || 'note') as SourceType;
    const matchedTopics = allTopics.filter(t => t.note_ids.includes(chunk.note_id));

    if (sourceType === 'note') {
      const note = noteMap.get(chunk.note_id);
      if (!note) continue;
      results.push({ note, source_type: 'note', chunk, score: hit.score, matchedTopics });
    } else if (sourceType === 'document') {
      const doc = docMap.get(chunk.note_id);
      if (!doc) continue;
      const syntheticNote: Note = {
        id: doc.id, title: doc.filename, content: '', plain_text: doc.plain_text,
        folder: 'Documents', tags: doc.tags, created_at: doc.created_at,
        updated_at: doc.updated_at, chunk_count: doc.chunk_count,
      };
      results.push({ note: syntheticNote, document: doc, source_type: 'document', chunk, score: hit.score, matchedTopics });
    } else if (sourceType === 'image') {
      const img = imageMap.get(chunk.note_id);
      if (!img) continue;
      const syntheticNote: Note = {
        id: img.id, title: img.filename, content: '', plain_text: img.ocr_text,
        folder: 'Images', tags: img.tags, created_at: img.created_at,
        updated_at: img.updated_at, chunk_count: img.chunk_count,
      };
      results.push({ note: syntheticNote, image: img, source_type: 'image', chunk, score: hit.score, matchedTopics });
    }

    if (results.length >= topK) break;
  }

  return results;
}

export async function findRelatedNotes(noteId: string, topK = 8): Promise<RelatedNote[]> {
  await initPipeline();
  const noteEmbs = db.getAllNoteEmbeddings();
  const current = noteEmbs.find(e => e.noteId === noteId);
  if (!current) return [];

  const allNotes = db.getAllNotes();
  const noteMap = new Map(allNotes.map(n => [n.id, n]));

  const allDocs = db.getAllDocuments();
  for (const doc of allDocs) {
    noteMap.set(doc.id, {
      id: doc.id, title: doc.filename, content: '', plain_text: doc.plain_text,
      folder: 'Documents', tags: doc.tags, created_at: doc.created_at,
      updated_at: doc.updated_at, chunk_count: doc.chunk_count,
    });
  }
  const allImages = db.getAllImages();
  for (const img of allImages) {
    noteMap.set(img.id, {
      id: img.id, title: img.filename, content: '', plain_text: img.ocr_text,
      folder: 'Images', tags: img.tags, created_at: img.created_at,
      updated_at: img.updated_at, chunk_count: img.chunk_count,
    });
  }

  const scored: { noteId: string; score: number }[] = [];
  for (const other of noteEmbs) {
    if (other.noteId === noteId) continue;
    const sim = cosineSimilarity(current.vector, other.vector);
    if (sim > 0.2) scored.push({ noteId: other.noteId, score: sim });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map(s => ({
    note: noteMap.get(s.noteId)!,
    score: s.score,
  })).filter(r => r.note);
}

export function summarize(text: string, maxSentences = 5): AISummary {
  const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 15);
  if (sentences.length <= maxSentences) {
    return { summary: sentences.join(' '), keyPoints: sentences };
  }

  const wordFreq = buildWordFreq(text);
  const scored = sentences.map((s, i) => ({
    text: s.trim(),
    index: i,
    score: scoreSentence(s, wordFreq, i, sentences.length),
  }));

  const topSentences = scored
    .sort((a, b) => b.score - a.score)
    .slice(0, maxSentences)
    .sort((a, b) => a.index - b.index);

  const summary = topSentences.map(s => s.text).join(' ');

  const indicators = ['important', 'key', 'main', 'essential', 'crucial', 'therefore', 'conclusion', 'definition', 'means'];
  const keyPoints = scored
    .filter(s => indicators.some(ind => s.text.toLowerCase().includes(ind)) || s.score > scored[Math.floor(scored.length * 0.2)]?.score)
    .sort((a, b) => b.score - a.score)
    .slice(0, 7)
    .sort((a, b) => a.index - b.index)
    .map(s => s.text);

  return { summary, keyPoints: keyPoints.length ? keyPoints : topSentences.map(s => s.text) };
}

const STOP_WORDS = new Set([
  'the','be','to','of','and','a','in','that','have','i','it','for','not','on','with',
  'he','as','you','do','at','this','but','his','by','from','they','we','say','her',
  'she','or','an','will','my','one','all','would','there','their','what','so','up',
  'out','if','about','who','get','which','go','me','when','make','can','like','time',
  'no','just','him','know','take','people','into','year','your','good','some','could',
  'them','see','other','than','then','now','look','only','come','its','over','think',
  'also','back','after','use','two','how','our','work','first','well','way','even',
  'new','want','because','any','these','give','day','most','us','are','is','was',
  'were','been','being','has','had','did','does','doing','shall','should','may',
  'might','must','need','used','using','each','every','both','few','more','many',
  'such','very','own','same','still','where','why','here','much','through','between',
]);

function extractKeywords(text: string): string[] {
  const words = text.toLowerCase().match(/\b[a-z][a-z'-]*[a-z]\b/g) || [];
  const freq: Record<string, number> = {};
  let total = 0;
  for (const w of words) {
    if (w.length < 3 || STOP_WORDS.has(w)) continue;
    freq[w] = (freq[w] || 0) + 1;
    total++;
  }
  if (total === 0) return [];

  return Object.entries(freq)
    .map(([word, count]) => ({ word, score: (count / total) * (1 + Math.min(word.length / 10, 1)) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 8)
    .map(s => s.word);
}

let rebuildTimer: ReturnType<typeof setTimeout> | null = null;
let rebuildInProgress = false;

function scheduleRebuildTopicsAndLinks(): void {
  if (rebuildTimer) clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(() => {
    rebuildTopicsAndLinksInternal().catch(err =>
      console.error('[AI] rebuildTopicsAndLinks error:', err)
    );
  }, 5000);
}

async function rebuildTopicsAndLinksInternal(): Promise<void> {
  if (rebuildInProgress) return;
  rebuildInProgress = true;
  try {
    await rebuildTopicsAndLinks();
  } finally {
    rebuildInProgress = false;
  }
}

async function rebuildTopicsAndLinks(): Promise<void> {
  const notes = db.getAllNotes();
  const docs = db.getAllDocuments();
  const imgs = db.getAllImages();

  type SourceItem = { id: string; text: string };
  const allItems: SourceItem[] = [
    ...notes.map(n => ({ id: n.id, text: n.plain_text || stripHtml(n.content) })),
    ...docs.map(d => ({ id: d.id, text: d.plain_text })),
    ...imgs.map(i => ({ id: i.id, text: i.ocr_text })),
  ];

  if (allItems.length < 1) return;

  const globalFreq: Record<string, { count: number; noteIds: Set<string> }> = {};

  for (const item of allItems) {
    const kws = extractKeywords(item.text);
    for (const kw of kws) {
      if (!globalFreq[kw]) globalFreq[kw] = { count: 0, noteIds: new Set() };
      globalFreq[kw].count++;
      globalFreq[kw].noteIds.add(item.id);
    }
  }

  const topWords = Object.entries(globalFreq)
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 50);

  db.deleteAllTopics();

  for (const [word, data] of topWords) {
    if (data.count < 1) continue;
    db.insertTopic({
      id: randomUUID(),
      name: word.charAt(0).toUpperCase() + word.slice(1),
      keywords: [word],
      noteIds: Array.from(data.noteIds),
      frequency: data.count,
    });
  }

  db.deleteAllLinks();
  const embs = db.getAllNoteEmbeddings();
  if (embs.length < 2) return;

  for (let i = 0; i < embs.length; i++) {
    for (let j = i + 1; j < embs.length; j++) {
      const sim = cosineSimilarity(embs[i].vector, embs[j].vector);
      if (sim > 0.35) {
        db.insertLink(embs[i].noteId, embs[j].noteId, sim);
      }
    }
  }
}

function buildWordFreq(text: string): Record<string, number> {
  const words = text.toLowerCase().match(/\b[a-z][a-z'-]*[a-z]\b/g) || [];
  const freq: Record<string, number> = {};
  for (const w of words) {
    if (!STOP_WORDS.has(w) && w.length >= 3) freq[w] = (freq[w] || 0) + 1;
  }
  return freq;
}

function scoreSentence(sentence: string, wordFreq: Record<string, number>, position: number, total: number): number {
  const words = sentence.toLowerCase().match(/\b[a-z][a-z'-]*[a-z]\b/g) || [];
  if (!words.length) return 0;

  let tfScore = 0;
  for (const w of words) if (wordFreq[w]) tfScore += wordFreq[w];
  tfScore /= words.length;

  let posScore = 1;
  if (position === 0) posScore = 1.5;
  else if (position === total - 1) posScore = 1.3;
  else if (position < total * 0.2) posScore = 1.2;

  const len = words.length;
  const lenScore = len < 5 ? 0.5 : len > 40 ? 0.8 : len <= 25 ? 1.2 : 1;

  return tfScore * posScore * lenScore;
}

export async function retrieveChunksForSummarization(
  sourceId: string | null,
  fullText: string,
  maxChunks = 10,
): Promise<string[]> {
  await initPipeline();

  let storedChunks: { id: string; content: string }[] = [];
  if (sourceId) {
    storedChunks = db.getChunksByNote(sourceId);
  }

  if (storedChunks.length === 0 && fullText) {
    const freshChunks = chunkText(fullText);
    if (freshChunks.length <= maxChunks) {
      return freshChunks.map(c => c.content);
    }
    return selectRepresentativeChunks(freshChunks.map(c => c.content), maxChunks);
  }

  if (storedChunks.length === 0) return [cleanInputText(fullText).slice(0, 6000)];

  if (storedChunks.length <= maxChunks) {
    return storedChunks.map(c => c.content);
  }

  const chunkEmbeddings = db.getEmbeddingsByChunkIds(storedChunks.map(c => c.id));
  const embMap = new Map(chunkEmbeddings.map(e => [e.chunkId, e.vector]));

  const withEmb = storedChunks
    .filter(c => embMap.has(c.id))
    .map(c => ({ content: c.content, embedding: embMap.get(c.id)! }));

  if (withEmb.length >= maxChunks) {
    const dim = withEmb[0].embedding.length;
    const mean = new Array(dim).fill(0);
    for (const ce of withEmb) {
      for (let i = 0; i < dim; i++) mean[i] += ce.embedding[i];
    }
    for (let i = 0; i < dim; i++) mean[i] /= withEmb.length;

    const scored = withEmb.map(ce => ({
      content: ce.content,
      score: cosineSimilarity(ce.embedding, mean),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, maxChunks).map(s => s.content);
  }

  return selectRepresentativeChunks(storedChunks.map(c => c.content), maxChunks);
}

function selectRepresentativeChunks(chunks: string[], maxChunks: number): string[] {
  if (chunks.length <= maxChunks) return chunks;

  const selected: string[] = [chunks[0]];
  const remaining = maxChunks - 2;
  const step = (chunks.length - 2) / (remaining + 1);

  for (let i = 1; i <= remaining; i++) {
    const idx = Math.round(i * step);
    if (idx > 0 && idx < chunks.length - 1) {
      selected.push(chunks[idx]);
    }
  }

  selected.push(chunks[chunks.length - 1]);
  return selected;
}

const RETRIEVAL_CANDIDATES = 20;
const RETRIEVAL_TOP_K = 3;
const CONTEXT_EXPAND_WORDS = 150;

export async function retrieveChunksForChat(
  sourceId: string | null,
  query: string,
  topK = 5,
): Promise<string[]> {
  const result = await contextualRetrieveForChat(sourceId, query, topK);
  return result.contexts;
}

const FILLER_PREFIXES = /^(can you |could you |please |I want to know |I'd like to know |tell me (about )?|help me understand |what is |what are |what's |do you know |I need to know )/i;
const PRONOUN_REFERENCES = /\b(this|that|it)\s+(mean|means|refer|refers|do|does|is|are|was|were)\b/gi;

function rewriteQueryForRetrieval(query: string): string {
  let q = query.trim();
  q = q.replace(FILLER_PREFIXES, '');
  q = q.replace(PRONOUN_REFERENCES, 'the referenced concept $2');
  q = q.trim();
  return q.length >= 3 ? q : query.trim();
}

function expandQueryTerms(sourceId: string, query: string): string[] {
  const queryTerms = extractSearchTerms(query);
  if (queryTerms.length === 0) return [];

  const chunks = db.getChunksByNote(sourceId);
  if (chunks.length === 0) return [];

  const matchingChunks = chunks.filter(c => {
    const lower = c.content.toLowerCase();
    return queryTerms.some(t => lower.includes(t));
  });
  if (matchingChunks.length === 0) return [];

  const coTerms: Record<string, number> = {};
  for (const chunk of matchingChunks) {
    const words = chunk.content.toLowerCase().match(/\b[a-z][a-z'-]*[a-z]\b/g) || [];
    for (const w of words) {
      if (w.length < 3 || STOP_WORDS.has(w) || queryTerms.includes(w)) continue;
      coTerms[w] = (coTerms[w] || 0) + 1;
    }
  }

  return Object.entries(coTerms)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([term]) => term);
}

function extractSearchTerms(query: string): string[] {
  const words = query.toLowerCase().match(/\b[a-z][a-z'-]*[a-z]\b/g) || [];
  const quoted = query.match(/"([^"]+)"/g)?.map(q => q.slice(1, -1).toLowerCase()) || [];
  const meaningful = words.filter(w => w.length >= 3 && !STOP_WORDS.has(w));
  return [...new Set([...quoted, ...meaningful])];
}

function getSourceText(sourceId: string): string | null {
  const note = db.getNote(sourceId);
  if (note) return note.plain_text || stripHtml(note.content);
  const doc = db.getDocument(sourceId);
  if (doc) return doc.plain_text;
  const img = db.getImage(sourceId);
  if (img) return img.ocr_text;
  return null;
}

interface RetrievalCandidate {
  chunkId: string;
  content: string;
  sourceId: string;
  startOffset: number;
  endOffset: number;
  semanticScore: number;
  keywordScore: number;
  combinedScore: number;
}

async function hybridRetrieve(
  sourceId: string | null,
  query: string,
  expandedTerms: string[],
  topK: number,
): Promise<RetrievalCandidate[]> {
  const queryEmb = await generateEmbedding(query);
  const candidates: RetrievalCandidate[] = [];
  const queryTerms = extractSearchTerms(query);
  const allTerms = [...new Set([...queryTerms, ...expandedTerms])];

  if (sourceId) {
    const storedChunks = db.getChunksByNote(sourceId);
    const chunkEmbeddings = db.getEmbeddingsByChunkIds(storedChunks.map(c => c.id));
    const embMap = new Map(chunkEmbeddings.map(e => [e.chunkId, e.vector]));

    for (const chunk of storedChunks) {
      const emb = embMap.get(chunk.id);
      const semanticScore = emb ? cosineSimilarity(queryEmb, emb) : 0;
      const lowerContent = chunk.content.toLowerCase();
      let kwHits = 0;
      for (const term of allTerms) {
        if (lowerContent.includes(term)) kwHits++;
      }
      const keywordScore = allTerms.length > 0 ? kwHits / allTerms.length : 0;
      const combinedScore = 0.6 * semanticScore + 0.4 * keywordScore;

      candidates.push({
        chunkId: chunk.id,
        content: chunk.content,
        sourceId: chunk.note_id,
        startOffset: chunk.start_offset,
        endOffset: chunk.end_offset,
        semanticScore,
        keywordScore,
        combinedScore,
      });
    }
  } else {
    const hits = chunkIndex.search(queryEmb, topK * 3, 0.0);
    const hitChunks = db.getChunksByIds(hits.map(h => h.id));
    const scoreMap = new Map(hits.map(h => [h.id, h.score]));

    for (const chunk of hitChunks) {
      const semanticScore = scoreMap.get(chunk.id) || 0;
      const lowerContent = chunk.content.toLowerCase();
      let kwHits = 0;
      for (const term of allTerms) {
        if (lowerContent.includes(term)) kwHits++;
      }
      const keywordScore = allTerms.length > 0 ? kwHits / allTerms.length : 0;
      const combinedScore = 0.6 * semanticScore + 0.4 * keywordScore;

      candidates.push({
        chunkId: chunk.id,
        content: chunk.content,
        sourceId: chunk.note_id,
        startOffset: chunk.start_offset,
        endOffset: chunk.end_offset,
        semanticScore,
        keywordScore,
        combinedScore,
      });
    }
  }

  candidates.sort((a, b) => b.combinedScore - a.combinedScore);
  return candidates.slice(0, topK);
}

function rerankChunks(
  candidates: RetrievalCandidate[],
  query: string,
  expandedTerms: string[],
  topK: number,
): RetrievalCandidate[] {
  const queryTerms = extractSearchTerms(query);
  const allTerms = [...new Set([...queryTerms, ...expandedTerms])];

  for (const candidate of candidates) {
    const words = candidate.content.toLowerCase().split(/\s+/);

    let termHits = 0;
    for (const word of words) {
      for (const term of allTerms) {
        if (word.includes(term)) { termHits++; break; }
      }
    }
    const tfScore = words.length > 0 ? termHits / words.length : 0;

    const termPositions: number[] = [];
    for (let i = 0; i < words.length; i++) {
      for (const term of allTerms) {
        if (words[i].includes(term)) { termPositions.push(i); break; }
      }
    }
    let proximityScore = 0;
    if (termPositions.length > 1) {
      const span = termPositions[termPositions.length - 1] - termPositions[0] + 1;
      proximityScore = termPositions.length / span;
    } else if (termPositions.length === 1) {
      proximityScore = 0.3;
    }

    const positionScore = Math.max(0, 1 - candidate.startOffset / 50000) * 0.2;

    candidate.combinedScore =
      0.35 * candidate.semanticScore +
      0.25 * candidate.keywordScore +
      0.2 * tfScore +
      0.1 * proximityScore +
      0.1 * positionScore;
  }

  candidates.sort((a, b) => b.combinedScore - a.combinedScore);
  return candidates.slice(0, topK);
}

interface ExpandedContext {
  text: string;
  matchedTerm: string;
  wordOffset: number;
  snippet: string;
}

function expandChunkContexts(
  sourceId: string | null,
  chunks: RetrievalCandidate[],
): ExpandedContext[] {
  if (!sourceId || chunks.length === 0) {
    return chunks.map(c => ({
      text: c.content,
      matchedTerm: '',
      wordOffset: -1,
      snippet: c.content.slice(0, 150) + (c.content.length > 150 ? '...' : ''),
    }));
  }

  const fullText = getSourceText(sourceId);
  if (!fullText) {
    return chunks.map(c => ({
      text: c.content,
      matchedTerm: '',
      wordOffset: -1,
      snippet: c.content.slice(0, 150) + '...',
    }));
  }

  const fullWords = fullText.split(/\s+/).filter(w => w.length > 0);

  return chunks.map(chunk => {
    const textBefore = fullText.slice(0, Math.max(0, chunk.startOffset));
    const wordsBefore = textBefore.split(/\s+/).filter(w => w.length > 0).length;
    const chunkWordCount = countWords(chunk.content);

    const expandStart = Math.max(0, wordsBefore - CONTEXT_EXPAND_WORDS);
    const expandEnd = Math.min(fullWords.length, wordsBefore + chunkWordCount + CONTEXT_EXPAND_WORDS);
    const expandedText = fullWords.slice(expandStart, expandEnd).join(' ');

    const snippetStart = Math.max(0, wordsBefore - 20);
    const snippetEnd = Math.min(fullWords.length, wordsBefore + chunkWordCount + 20);
    const snippet = (snippetStart > 0 ? '...' : '') +
      fullWords.slice(snippetStart, snippetEnd).join(' ') +
      (snippetEnd < fullWords.length ? '...' : '');

    return { text: expandedText, matchedTerm: '', wordOffset: wordsBefore, snippet };
  });
}

function compressContext(
  expandedChunks: ExpandedContext[],
  query: string,
  expandedTerms: string[],
): ExpandedContext[] {
  const queryTerms = extractSearchTerms(query);
  const allTerms = [...new Set([...queryTerms, ...expandedTerms])];
  const globalSeen = new Set<string>();

  return expandedChunks.map(chunk => {
    const sentences = chunk.text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 5);

    const scoredSentences = sentences.map(s => {
      const lower = s.toLowerCase();
      let score = 0;
      for (const term of allTerms) {
        if (lower.includes(term)) score += 1;
      }
      return { text: s, score };
    });

    let relevant = scoredSentences.filter(s => s.score > 0);
    if (relevant.length === 0) relevant = scoredSentences;

    const deduped = relevant.filter(s => {
      const key = s.text.trim().toLowerCase().slice(0, 100);
      if (globalSeen.has(key)) return false;
      globalSeen.add(key);
      return true;
    });

    const compressed = deduped.map(s => s.text).join(' ');
    return {
      text: compressed || chunk.text,
      matchedTerm: chunk.matchedTerm,
      wordOffset: chunk.wordOffset,
      snippet: chunk.snippet,
    };
  });
}

export async function contextualRetrieveForChat(
  sourceId: string | null,
  query: string,
  _topK = 5,
): Promise<ChatRetrievalResult> {
  await initPipeline();

  const rewrittenQuery = rewriteQueryForRetrieval(query);

  const expandedTerms = sourceId ? expandQueryTerms(sourceId, rewrittenQuery) : [];

  console.log(`[AI] Pipeline: query="${query}" → rewritten="${rewrittenQuery}" expanded=[${expandedTerms.join(', ')}]`);

  const candidates = await hybridRetrieve(sourceId, rewrittenQuery, expandedTerms, RETRIEVAL_CANDIDATES);

  if (candidates.length > 0) {
    const reranked = rerankChunks([...candidates], rewrittenQuery, expandedTerms, RETRIEVAL_TOP_K);

    const expanded = expandChunkContexts(sourceId, reranked);

    const compressed = compressContext(expanded, rewrittenQuery, expandedTerms);

    const queryTerms = extractSearchTerms(query);
    const sources: ContextualMatch[] = compressed.map((c, i) => ({
      context: c.text,
      matchedTerm: reranked[i]
        ? queryTerms.find(t => reranked[i].content.toLowerCase().includes(t)) || ''
        : '',
      wordOffset: c.wordOffset,
      snippet: c.snippet,
    }));

    console.log(`[AI] Pipeline: ${candidates.length} candidates → ${reranked.length} reranked → ${compressed.length} contexts`);
    return { contexts: compressed.map(c => c.text), sources, method: 'contextual' };
  }

  if (sourceId) {
    const fullText = getSourceText(sourceId);
    if (fullText && fullText.trim().length > 0) {
      const freshChunks = chunkText(fullText);
      const queryTerms = extractSearchTerms(rewrittenQuery);
      const allTerms = [...new Set([...queryTerms, ...expandedTerms])];

      const scored = freshChunks
        .map(c => {
          const lower = c.content.toLowerCase();
          let kwScore = 0;
          for (const term of allTerms) {
            if (lower.includes(term)) kwScore++;
          }
          return { ...c, score: kwScore };
        })
        .filter(c => c.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, RETRIEVAL_TOP_K);

      if (scored.length > 0) {
        return {
          contexts: scored.map(c => c.content),
          sources: scored.map(c => ({
            context: c.content,
            matchedTerm: allTerms.find(t => c.content.toLowerCase().includes(t)) || '',
            wordOffset: -1,
            snippet: c.content.slice(0, 150) + (c.content.length > 150 ? '...' : ''),
          })),
          method: 'contextual',
        };
      }
    }
  }

  return { contexts: [], sources: [], method: 'fulltext' };
}
