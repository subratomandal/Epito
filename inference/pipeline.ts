import { randomUUID } from 'crypto';
import * as db from '@/notes/database';
import { generateEmbedding, initEmbeddings, EMBEDDING_DIM } from '@/memory/embeddings';
import { VectorIndex, cosineSimilarity } from '@/memory/vector';
import { cleanInputText } from '@/model/llm';
import type { Note, SearchResult, RelatedNote, AISummary, SourceType, ContextualMatch, ChatRetrievalResult } from '@/common/types';
import { stripHtml } from '@/common/utils';

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
const CHUNK_OVERLAP_WORDS = 125; // 25% overlap (was 75/15%)

function countWords(text: string): number {
  return text.split(/\s+/).filter(w => w.length > 0).length;
}

function splitSentences(text: string): string[] {
  const raw = text.split(/(?<=[.!?])\s+|(?:\n\s*\n)/);
  return raw.map(s => s.trim()).filter(s => s.length > 0);
}

// --- Frontmatter Stripping

function stripFrontmatter(text: string): string {
  return text.replace(/^---\s*\n[\s\S]*?\n---\s*\n/, '').trim();
}

// --- Structure-Aware Pre-Processing
// Extracts code blocks and tables before sentence splitting; preserved as single chunks.

interface StructuredBlock {
  type: 'prose' | 'code' | 'table';
  content: string;
  startOffset: number;
}

function extractStructuredBlocks(text: string): StructuredBlock[] {
  const blocks: StructuredBlock[] = [];

  // Regex for fenced code blocks (```...```)
  const codeBlockRe = /```[\s\S]*?```/g;
  // Regex for markdown tables (lines with |)
  const tableRe = /(?:^|\n)(\|.+\|(?:\n\|[-:| ]+\|)?(?:\n\|.+\|)+)/gm;

  // Collect all structured regions
  const regions: { start: number; end: number; type: 'code' | 'table'; content: string }[] = [];

  let m;
  while ((m = codeBlockRe.exec(text)) !== null) {
    regions.push({ start: m.index, end: m.index + m[0].length, type: 'code', content: m[0] });
  }
  while ((m = tableRe.exec(text)) !== null) {
    const s = m.index + (m[0].startsWith('\n') ? 1 : 0);
    regions.push({ start: s, end: m.index + m[0].length, type: 'table', content: m[0].trim() });
  }

  // Sort by position, no overlaps
  regions.sort((a, b) => a.start - b.start);
  const merged: typeof regions = [];
  for (const r of regions) {
    if (merged.length > 0 && r.start < merged[merged.length - 1].end) continue;
    merged.push(r);
  }

  // Split text into prose and structured blocks
  let cursor = 0;
  for (const region of merged) {
    if (region.start > cursor) {
      const prose = text.slice(cursor, region.start).trim();
      if (prose) blocks.push({ type: 'prose', content: prose, startOffset: cursor });
    }
    blocks.push({ type: region.type, content: region.content, startOffset: region.start });
    cursor = region.end;
  }
  if (cursor < text.length) {
    const prose = text.slice(cursor).trim();
    if (prose) blocks.push({ type: 'prose', content: prose, startOffset: cursor });
  }

  return blocks.length > 0 ? blocks : [{ type: 'prose', content: text, startOffset: 0 }];
}

function chunkText(text: string): { content: string; startOffset: number; endOffset: number }[] {
  const cleaned = cleanInputText(stripFrontmatter(text));
  if (!cleaned.trim()) return [];

  const blocks = extractStructuredBlocks(cleaned);
  const allChunks: { content: string; startOffset: number; endOffset: number }[] = [];

  for (const block of blocks) {
    if (block.type === 'code' || block.type === 'table') {
      // Structured blocks are NEVER split — kept as single chunks
      allChunks.push({
        content: block.content,
        startOffset: block.startOffset,
        endOffset: block.startOffset + block.content.length,
      });
      continue;
    }

    // Split on paragraph boundaries first to preserve entity relationships.
    const paragraphs = block.content.split(/\n\s*\n/).filter(p => p.trim().length > 0);

    let currentParagraphs: string[] = [];
    let currentWords = 0;
    let offset = block.startOffset;
    let chunkStart = block.startOffset;

    for (const paragraph of paragraphs) {
      const paraWords = countWords(paragraph);

      // If this single paragraph exceeds max, split it by sentences
      if (paraWords > CHUNK_MAX_WORDS) {
        // Flush current buffer first
        if (currentParagraphs.length > 0) {
          const content = currentParagraphs.join('\n\n');
          allChunks.push({ content, startOffset: chunkStart, endOffset: offset });
          currentParagraphs = [];
          currentWords = 0;
          chunkStart = offset;
        }

        // Split large paragraph by sentences with overlap
        const sentences = splitSentences(paragraph);
        let sentBuf: string[] = [];
        let sentWords = 0;
        let sentStart = offset;

        for (const sentence of sentences) {
          const sw = countWords(sentence);
          if (sentWords + sw > CHUNK_MAX_WORDS && sentWords >= CHUNK_MIN_WORDS) {
            allChunks.push({
              content: sentBuf.join(' '),
              startOffset: sentStart,
              endOffset: offset,
            });
            // Overlap: keep last ~25% of sentences
            let overlapWords = 0;
            let overlapIdx = sentBuf.length;
            for (let j = sentBuf.length - 1; j >= 0; j--) {
              const w = countWords(sentBuf[j]);
              if (overlapWords + w > CHUNK_OVERLAP_WORDS) break;
              overlapWords += w;
              overlapIdx = j;
            }
            sentBuf = sentBuf.slice(overlapIdx);
            sentWords = overlapWords;
            sentStart = Math.max(block.startOffset, offset - sentBuf.join(' ').length);
          }
          sentBuf.push(sentence);
          sentWords += sw;
          offset += sentence.length + 1;
        }
        if (sentBuf.length > 0) {
          allChunks.push({ content: sentBuf.join(' '), startOffset: sentStart, endOffset: offset });
        }
        chunkStart = offset;
        continue;
      }

      // Would adding this paragraph exceed the limit?
      if (currentWords + paraWords > CHUNK_MAX_WORDS && currentWords >= CHUNK_MIN_WORDS) {
        const content = currentParagraphs.join('\n\n');
        allChunks.push({ content, startOffset: chunkStart, endOffset: offset });

        // Overlap: keep last paragraph if small enough
        const lastPara = currentParagraphs[currentParagraphs.length - 1];
        const lastWords = countWords(lastPara || '');
        if (lastWords <= CHUNK_OVERLAP_WORDS && currentParagraphs.length > 1) {
          currentParagraphs = [lastPara];
          currentWords = lastWords;
          chunkStart = Math.max(block.startOffset, offset - lastPara.length);
        } else {
          currentParagraphs = [];
          currentWords = 0;
          chunkStart = offset;
        }
      }

      currentParagraphs.push(paragraph);
      currentWords += paraWords;
      offset += paragraph.length + 2; // +2 for \n\n
    }

    // Flush remaining
    if (currentParagraphs.length > 0) {
      const content = currentParagraphs.join('\n\n');
      if (currentWords < CHUNK_MIN_WORDS / 2 && allChunks.length > 0) {
        const last = allChunks[allChunks.length - 1];
        allChunks[allChunks.length - 1] = {
          content: last.content + '\n\n' + content,
          startOffset: last.startOffset,
          endOffset: offset,
        };
      } else {
        allChunks.push({ content, startOffset: chunkStart, endOffset: offset });
      }
    }
  }

  return allChunks.length > 0 ? allChunks : [{ content: cleaned, startOffset: 0, endOffset: cleaned.length }];
}

// --- Entity Extraction

const ENTITY_EXTRACTORS: { type: string; patterns: RegExp[] }[] = [
  {
    type: 'ORG',
    patterns: [
      /(?:General\s+(?:Sir\s+)?)?[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|and|for|in|Sir|General))*\s+(?:University|Institute|College|Academy|School)/g,
      /University\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g,
      /[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Co|Group|Foundation|Technologies|Labs|Systems)\.?)/g,
    ],
  },
  {
    type: 'PERSON',
    patterns: [
      /[A-Z][a-z]{1,15}\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?/g,
    ],
  },
  {
    type: 'TOOL',
    patterns: [
      /\b(?:BeautifulSoup|Scrapy|Selenium|lxml|Flask|Django|MongoDB|PostgreSQL|MySQL|Redis|TensorFlow|PyTorch|Keras|NumPy|Pandas|Scikit-learn|OpenCV|Docker|Kubernetes|React|Angular|Vue|Next\.js|Node\.js)\b/g,
    ],
  },
];

function extractAndStoreEntities(
  documentId: string,
  chunks: { content: string; startOffset: number; endOffset: number }[],
  chunkIds: string[],
): void {
  const seen = new Set<string>();

  for (let i = 0; i < chunks.length; i++) {
    const text = chunks[i].content;
    const chunkId = chunkIds[i];

    for (const { type, patterns } of ENTITY_EXTRACTORS) {
      for (const pattern of patterns) {
        pattern.lastIndex = 0;
        let m;
        while ((m = pattern.exec(text)) !== null) {
          let name = m[0].trim();
          if (name.length < 3) continue;

          // Normalize common abbreviations
          if (name === 'MIT') name = 'Massachusetts Institute of Technology';

          const key = `${type}:${name.toLowerCase()}`;
          if (seen.has(key)) continue;
          seen.add(key);

          // Extract evidence sentence
          const sentStart = Math.max(0, text.lastIndexOf('.', m.index) + 1);
          const sentEnd = text.indexOf('.', m.index + m[0].length);
          const evidence = text.slice(sentStart, sentEnd > 0 ? sentEnd + 1 : m.index + m[0].length + 80).trim().slice(0, 200);

          db.insertEntity(documentId, name, type, chunkId, evidence);
        }
      }
    }
  }

  console.log(`[AI] Entity index: ${seen.size} entities extracted for document ${documentId}`);
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

  // Build entity index — extract organizations, people, tools from all chunks
  db.deleteEntitiesByDocument(noteId);
  extractAndStoreEntities(noteId, chunks, chunkIds);
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

const RETRIEVAL_CANDIDATES = 30;
const RETRIEVAL_TOP_K = 6;
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

// --- BM25-style keyword scoring (TF normalized by document length)

function bm25Score(content: string, terms: string[], k1 = 1.2, b = 0.75, avgDl = 200): number {
  const words = content.toLowerCase().split(/\s+/);
  const dl = words.length;
  let score = 0;

  for (const term of terms) {
    // Term frequency in this document
    let tf = 0;
    for (const w of words) {
      if (w.includes(term)) tf++;
    }
    if (tf === 0) continue;

    // BM25 TF component (saturating term frequency)
    const tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgDl)));
    score += tfNorm;
  }

  return terms.length > 0 ? score / terms.length : 0;
}

async function hybridRetrieve(
  sourceId: string | null,
  query: string,
  expandedTerms: string[],
  topK: number,
): Promise<RetrievalCandidate[]> {
  const queryEmb = await generateEmbedding(query);
  const queryTerms = extractSearchTerms(query);
  const allTerms = [...new Set([...queryTerms, ...expandedTerms])];

  // Collect candidates with separate semantic and keyword scores
  const rawCandidates: Array<{
    chunkId: string; content: string; sourceId: string;
    startOffset: number; endOffset: number;
    semanticScore: number; keywordScore: number;
  }> = [];

  if (sourceId) {
    const storedChunks = db.getChunksByNote(sourceId);
    const chunkEmbeddings = db.getEmbeddingsByChunkIds(storedChunks.map(c => c.id));
    const embMap = new Map(chunkEmbeddings.map(e => [e.chunkId, e.vector]));

    for (const chunk of storedChunks) {
      const emb = embMap.get(chunk.id);
      rawCandidates.push({
        chunkId: chunk.id, content: chunk.content, sourceId: chunk.note_id,
        startOffset: chunk.start_offset, endOffset: chunk.end_offset,
        semanticScore: emb ? cosineSimilarity(queryEmb, emb) : 0,
        keywordScore: bm25Score(chunk.content, allTerms),
      });
    }
  } else {
    const hits = chunkIndex.search(queryEmb, topK * 3, 0.0);
    const hitChunks = db.getChunksByIds(hits.map(h => h.id));
    const scoreMap = new Map(hits.map(h => [h.id, h.score]));

    for (const chunk of hitChunks) {
      rawCandidates.push({
        chunkId: chunk.id, content: chunk.content, sourceId: chunk.note_id,
        startOffset: chunk.start_offset, endOffset: chunk.end_offset,
        semanticScore: scoreMap.get(chunk.id) || 0,
        keywordScore: bm25Score(chunk.content, allTerms),
      });
    }
  }

  // Reciprocal Rank Fusion: combine rank positions instead of raw scores (k=60 standard)
  const RRF_K = 60;

  // Rank by semantic score
  const semRanked = [...rawCandidates].sort((a, b) => b.semanticScore - a.semanticScore);
  const semRankMap = new Map<string, number>();
  semRanked.forEach((c, i) => semRankMap.set(c.chunkId, i + 1));

  // Rank by keyword/BM25 score
  const kwRanked = [...rawCandidates].sort((a, b) => b.keywordScore - a.keywordScore);
  const kwRankMap = new Map<string, number>();
  kwRanked.forEach((c, i) => kwRankMap.set(c.chunkId, i + 1));

  // Compute RRF combined score
  const candidates: RetrievalCandidate[] = rawCandidates.map(c => {
    const semRank = semRankMap.get(c.chunkId) || rawCandidates.length;
    const kwRank = kwRankMap.get(c.chunkId) || rawCandidates.length;
    const rrfScore = 1 / (RRF_K + semRank) + 1 / (RRF_K + kwRank);

    return {
      ...c,
      combinedScore: rrfScore,
    };
  });

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

// --- MMR Reranking: penalize candidates too similar to already-selected chunks for diversity

async function mmrRerank(
  candidates: RetrievalCandidate[],
  topK: number,
  lambda: number = 0.7, // 0.7 = favor relevance, 0.5 = balance diversity
): Promise<RetrievalCandidate[]> {
  if (candidates.length <= topK) return candidates;

  // Get embeddings for all candidates
  const embMap = new Map<string, number[]>();
  const storedEmbs = db.getEmbeddingsByChunkIds(candidates.map(c => c.chunkId));
  for (const e of storedEmbs) embMap.set(e.chunkId, e.vector);

  const selected: RetrievalCandidate[] = [];
  const remaining = [...candidates];

  while (selected.length < topK && remaining.length > 0) {
    let bestIdx = 0;
    let bestScore = -Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const cand = remaining[i];
      const relevance = cand.combinedScore;

      // Max similarity to any already-selected chunk
      let maxSim = 0;
      if (selected.length > 0) {
        const candEmb = embMap.get(cand.chunkId);
        if (candEmb) {
          for (const sel of selected) {
            const selEmb = embMap.get(sel.chunkId);
            if (selEmb) {
              const sim = cosineSimilarity(candEmb, selEmb);
              if (sim > maxSim) maxSim = sim;
            }
          }
        }
      }

      const mmrScore = lambda * relevance - (1 - lambda) * maxSim;
      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIdx = i;
      }
    }

    selected.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }

  return selected;
}

// --- Adjacent Chunk Expansion: include N-1 and N+1 to handle boundary-spanning answers

function expandWithAdjacentChunks(
  selected: RetrievalCandidate[],
  allChunksForSource: Map<string, RetrievalCandidate[]>,
): RetrievalCandidate[] {
  const resultIds = new Set(selected.map(c => c.chunkId));
  const expanded: RetrievalCandidate[] = [...selected];

  for (const chunk of selected) {
    const sourceChunks = allChunksForSource.get(chunk.sourceId);
    if (!sourceChunks) continue;

    // Sort source chunks by offset to find neighbors
    const sorted = [...sourceChunks].sort((a, b) => a.startOffset - b.startOffset);
    const idx = sorted.findIndex(c => c.chunkId === chunk.chunkId);
    if (idx < 0) continue;

    // Add N-1
    if (idx > 0 && !resultIds.has(sorted[idx - 1].chunkId)) {
      const prev = sorted[idx - 1];
      resultIds.add(prev.chunkId);
      expanded.push({ ...prev, combinedScore: prev.combinedScore * 0.7 }); // lower score for context chunks
    }
    // Add N+1
    if (idx < sorted.length - 1 && !resultIds.has(sorted[idx + 1].chunkId)) {
      const next = sorted[idx + 1];
      resultIds.add(next.chunkId);
      expanded.push({ ...next, combinedScore: next.combinedScore * 0.7 });
    }
  }

  // Re-sort by offset within each source to maintain reading order
  expanded.sort((a, b) => {
    if (a.sourceId !== b.sourceId) return a.sourceId.localeCompare(b.sourceId);
    return a.startOffset - b.startOffset;
  });

  return expanded;
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

    // Keep ALL sentences — don't strip non-matching ones.
    // Stripping causes answer sentences with paraphrased/implicit information
    // to be lost before the model ever sees them. Sort by relevance but retain all.
    const sorted = [...scoredSentences].sort((a, b) => b.score - a.score);

    const deduped = sorted.filter(s => {
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
    const reranked = rerankChunks([...candidates], rewrittenQuery, expandedTerms, RETRIEVAL_TOP_K * 2);

    // MMR diversity enforcement — prevents all top-K from the same section
    const mmrSelected = await mmrRerank(reranked, RETRIEVAL_TOP_K, 0.7);

    // Adjacent chunk expansion — include N-1 and N+1 for boundary coverage
    const sourceChunkMap = new Map<string, RetrievalCandidate[]>();
    for (const c of candidates) {
      if (!sourceChunkMap.has(c.sourceId)) sourceChunkMap.set(c.sourceId, []);
      sourceChunkMap.get(c.sourceId)!.push(c);
    }
    const withAdjacent = expandWithAdjacentChunks(mmrSelected, sourceChunkMap);

    const expanded = expandChunkContexts(sourceId, withAdjacent.slice(0, RETRIEVAL_TOP_K + 4));

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
