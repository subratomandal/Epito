'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { ScrollArea, Badge, Separator, Tooltip } from '@/components/ui/primitives';
import {
  RefreshCw, Copy, Check, Send,
  PanelRightClose, PanelRight,
} from 'lucide-react';
import { stripHtml } from '@/common/utils';

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
  'while','before','those','right','down','long','made','found','called','part',
  'may','said','since','however','within','without','another','point','around',
]);

function extractTopics(text: string): string[] {
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
    .map(([word, count]) => ({
      word: word.charAt(0).toUpperCase() + word.slice(1),
      score: (count / total) * (1 + Math.min(word.length / 10, 1)),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 10)
    .map(s => s.word);
}

function cleanChatText(text: string): string {
  return text
    .replace(/<\/s>/g, '')
    .replace(/<\|im_end\|>/g, '')
    .replace(/<\|endoftext\|>/g, '')
    .replace(/<\|end\|>/g, '')
    .replace(/\[INST\][\s\S]*?\[\/INST\]/g, '')
    .replace(/\b(terminated|Terminated)\s*\.?\s*$/i, '')
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    .replace(/`{1,3}([^`]+)`{1,3}/g, '$1')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/[<>{}[\]\\|~^]/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

interface StructuredCategory {
  title: string;
  points: string[];
}

function parseStructuredSummary(text: string): StructuredCategory[] {
  const categories: StructuredCategory[] = [];
  let currentTitle = '';
  let currentPoints: string[] = [];

  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    if (/^[A-Z][A-Z\s]{2,}$/.test(trimmed)) {
      if (currentTitle && currentPoints.length > 0) {
        categories.push({ title: currentTitle, points: currentPoints });
      }
      currentTitle = trimmed;
      currentPoints = [];
    } else if (trimmed.startsWith('- ')) {
      currentPoints.push(trimmed.slice(2).trim());
    } else if (currentTitle) {
      currentPoints.push(trimmed);
    } else {
      if (!categories.length && !currentTitle) {
        currentTitle = 'OVERVIEW';
        currentPoints.push(trimmed);
      }
    }
  }

  if (currentTitle && currentPoints.length > 0) {
    categories.push({ title: currentTitle, points: currentPoints });
  }

  if (categories.length === 0 && text.trim()) {
    const points = text.split('\n').filter(l => l.trim()).map(l => l.replace(/^-\s*/, '').trim());
    categories.push({ title: 'SUMMARY', points });
  }

  return categories;
}

interface SentenceExplanation {
  text: string;
  explanation: string;
}

interface StreamRetrievalData {
  method: 'contextual' | 'embedding' | 'fulltext';
  sources: ChatSource[];
}

interface BlockSummaryEvent {
  summary: string;
  blockIndex: number;
  totalBlocks: number;
}

interface BlockStreamEvent {
  text: string;
  blockIndex: number;
  totalBlocks: number;
}

async function readStream(
  response: Response,
  onChunk: (text: string) => void,
  onProgress?: (msg: string) => void,
  onError?: (msg: string) => void,
  signal?: AbortSignal,
  onRetrieval?: (data: StreamRetrievalData) => void,
  onBlockSummary?: (event: BlockSummaryEvent) => void,
  onBlockStream?: (event: BlockStreamEvent) => void,
): Promise<string> {
  const reader = response.body?.getReader();
  if (!reader) return '';
  const decoder = new TextDecoder();
  let fullText = '';
  let buffer = '';

  try {
    while (true) {
      if (signal?.aborted) {
        reader.cancel();
        return fullText;
      }

      const { done, value } = await reader.read();
      if (done) break;

      if (signal?.aborted) {
        reader.cancel();
        return fullText;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data: ') && trimmed !== 'data: [DONE]') {
          try {
            const parsed = JSON.parse(trimmed.slice(6));
            if (parsed.error && onError) {
              onError(parsed.error);
            }
            if (parsed.text) {
              fullText = parsed.text;
              onChunk(fullText);
            }
            if (parsed.progress && onProgress) {
              onProgress(parsed.progress);
            }
            if (parsed.retrieval && onRetrieval) {
              onRetrieval(parsed.retrieval as StreamRetrievalData);
            }
            if (parsed.blockDone && onBlockSummary) {
              onBlockSummary({
                summary: parsed.blockDone,
                blockIndex: parsed.blockIndex ?? 0,
                totalBlocks: parsed.totalBlocks ?? 1,
              });
            }
            if (parsed.blockStream && onBlockStream) {
              onBlockStream({
                text: parsed.blockStream,
                blockIndex: parsed.blockIndex ?? 0,
                totalBlocks: parsed.totalBlocks ?? 1,
              });
            }
          } catch {}
        }
      }
    }
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      return fullText;
    }
    throw err;
  }
  return fullText;
}

function SkeletonBlock({ lines = 3, message }: { lines?: number; message?: string }) {
  return (
    <div className="py-1.5 space-y-1.5">
      {message && (
        <span className="text-[10px] text-muted-foreground/50">{message}</span>
      )}
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-[9px] rounded-sm bg-muted/50"
          style={{
            width: i === lines - 1 ? '55%' : i === 0 ? '90%' : '100%',
            animation: `skeletonPulse 1.6s ease-in-out ${i * 0.12}s infinite`,
          }}
        />
      ))}
      <style>{`
        @keyframes skeletonPulse {
          0%, 100% { opacity: 0.45; }
          50% { opacity: 0.15; }
        }
      `}</style>
    </div>
  );
}

interface ChatSource {
  matchedTerm: string;
  wordOffset: number;
  snippet: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
  retrievalMethod?: 'contextual' | 'embedding' | 'fulltext';
}

type CopiedField = 'summary' | 'explain' | 'chat' | null;
type ActiveMode = 'summarize' | 'explain' | 'chat' | null;

interface DocAIState {
  summaryPhase: 'idle' | 'preparing' | 'streaming' | 'section-done' | 'all-done' | 'merging' | 'merged';
  sectionChunks: string[];
  sectionCacheIds: string[];
  sectionSummaries: string[];
  currentSectionIdx: number;
  currentStreamText: string;
  mergedSummary: string;
  showAllPoints: Record<string, boolean>;
  explainPhase: 'idle' | 'preparing' | 'streaming' | 'section-done' | 'all-done';
  explainChunks: string[];
  explainCacheIds: string[];
  explainSections: SentenceExplanation[][];
  explainCurrentIdx: number;
  explainStreamText: string;
  explainSource: 'llm' | 'extractive' | '';
  chatMessages: ChatMessage[];
  chatInput: string;
  chatLoading: boolean;
  chatStreaming: boolean;
  activeMode: ActiveMode;
  copiedField: CopiedField;
  activeTopic: string | null;
  progressMsg: string;
}

function defaultDocState(): DocAIState {
  return {
    summaryPhase: 'idle',
    sectionChunks: [],
    sectionCacheIds: [],
    sectionSummaries: [],
    currentSectionIdx: -1,
    currentStreamText: '',
    mergedSummary: '',
    showAllPoints: {},
    explainPhase: 'idle',
    explainChunks: [],
    explainCacheIds: [],
    explainSections: [],
    explainCurrentIdx: -1,
    explainStreamText: '',
    explainSource: '',
    chatMessages: [],
    chatInput: '',
    chatLoading: false,
    chatStreaming: false,
    activeMode: null,
    copiedField: null,
    activeTopic: null,
    progressMsg: '',
  };
}

const DOC_CACHE_MAX = 20;

interface AIPanelProps {
  noteId: string | null;
  noteContent: string;
  onTopicClick?: (topic: string) => void;
  isMobile?: boolean;
  onClose?: () => void;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

export default function AIPanel({ noteId, noteContent, onTopicClick, isMobile, onClose, collapsed, onToggleCollapse }: AIPanelProps) {
  const [aiAvailable, setAiAvailable] = useState<boolean | null>(null);

  const [summaryPhase, setSummaryPhase] = useState<DocAIState['summaryPhase']>('idle');
  const [sectionChunks, setSectionChunks] = useState<string[]>([]);
  const [sectionCacheIds, setSectionCacheIds] = useState<string[]>([]);
  const [sectionSummaries, setSectionSummaries] = useState<string[]>([]);
  const [currentSectionIdx, setCurrentSectionIdx] = useState(-1);
  const [currentStreamText, setCurrentStreamText] = useState('');
  const [mergedSummary, setMergedSummary] = useState('');
  const [showAllPoints, setShowAllPoints] = useState<Record<string, boolean>>({});

  const [explainPhase, setExplainPhase] = useState<DocAIState['explainPhase']>('idle');
  const [explainChunks, setExplainChunks] = useState<string[]>([]);
  const [explainCacheIds, setExplainCacheIds] = useState<string[]>([]);
  const [explainSections, setExplainSections] = useState<SentenceExplanation[][]>([]);
  const [explainCurrentIdx, setExplainCurrentIdx] = useState(-1);
  const [explainStreamText, setExplainStreamText] = useState('');
  const [explainSource, setExplainSource] = useState<'llm' | 'extractive' | ''>('');

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatStreaming, setChatStreaming] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  const [activeMode, setActiveMode] = useState<ActiveMode>(null);
  const [panelWidth, setPanelWidth] = useState(340);
  const [isResizing, setIsResizing] = useState(false);
  const [copiedField, setCopiedField] = useState<CopiedField>(null);
  const [activeTopic, setActiveTopic] = useState<string | null>(null);
  const [progressMsg, setProgressMsg] = useState('');
  const resizingRef = useRef(false);
  const resizeStartRef = useRef({ x: 0, width: 0 });

  const docCacheRef = useRef<Map<string, DocAIState>>(new Map());
  const abortRef = useRef<AbortController>(new AbortController());
  const noteIdRef = useRef<string | null>(noteId);
  const prevNoteIdRef = useRef<string | null>(noteId);

  const getDocState = useCallback((id: string): DocAIState => {
    if (!docCacheRef.current.has(id)) {
      docCacheRef.current.set(id, defaultDocState());
    }
    return docCacheRef.current.get(id)!;
  }, []);

  const saveCurrentState = useCallback((docId: string) => {
    let stableSummaryPhase = summaryPhase;
    if (stableSummaryPhase === 'preparing' || stableSummaryPhase === 'streaming') {
      stableSummaryPhase = sectionSummaries.length > 0 ? 'section-done' : 'idle';
    }
    if (stableSummaryPhase === 'merging') {
      stableSummaryPhase = sectionSummaries.length > 1 ? 'all-done' : 'idle';
    }

    let stableExplainPhase = explainPhase;
    if (stableExplainPhase === 'preparing' || stableExplainPhase === 'streaming') {
      stableExplainPhase = explainSections.length > 0 ? 'section-done' : 'idle';
    }

    const state: DocAIState = {
      summaryPhase: stableSummaryPhase,
      sectionChunks,
      sectionCacheIds,
      sectionSummaries,
      currentSectionIdx,
      currentStreamText: '',
      mergedSummary,
      showAllPoints,
      explainPhase: stableExplainPhase,
      explainChunks,
      explainCacheIds,
      explainSections,
      explainCurrentIdx,
      explainStreamText: '',
      explainSource,
      chatMessages,
      chatInput,
      chatLoading: false,
      chatStreaming: false,
      activeMode,
      copiedField: null,
      activeTopic: null,
      progressMsg: '',
    };
    docCacheRef.current.set(docId, state);

    if (docCacheRef.current.size > DOC_CACHE_MAX) {
      const keys = Array.from(docCacheRef.current.keys());
      const toRemove = keys.slice(0, keys.length - DOC_CACHE_MAX);
      for (const key of toRemove) {
        docCacheRef.current.delete(key);
      }
    }
  }, [
    summaryPhase, sectionChunks, sectionCacheIds, sectionSummaries,
    currentSectionIdx, mergedSummary, showAllPoints,
    explainPhase, explainChunks, explainCacheIds, explainSections,
    explainCurrentIdx, explainSource,
    chatMessages, chatInput, activeMode,
  ]);

  const loadState = useCallback((state: DocAIState) => {
    setSummaryPhase(state.summaryPhase);
    setSectionChunks(state.sectionChunks);
    setSectionCacheIds(state.sectionCacheIds);
    setSectionSummaries(state.sectionSummaries);
    setCurrentSectionIdx(state.currentSectionIdx);
    setCurrentStreamText(state.currentStreamText);
    setMergedSummary(state.mergedSummary);
    setShowAllPoints(state.showAllPoints);
    setExplainPhase(state.explainPhase);
    setExplainChunks(state.explainChunks);
    setExplainCacheIds(state.explainCacheIds);
    setExplainSections(state.explainSections);
    setExplainCurrentIdx(state.explainCurrentIdx);
    setExplainStreamText(state.explainStreamText);
    setExplainSource(state.explainSource);
    setChatMessages(state.chatMessages);
    setChatInput(state.chatInput);
    setChatLoading(state.chatLoading);
    setChatStreaming(state.chatStreaming);
    setActiveMode(state.activeMode);
    setCopiedField(state.copiedField);
    setActiveTopic(state.activeTopic);
    setProgressMsg(state.progressMsg);
  }, []);

  const plainText = useMemo(() => {
    if (!noteContent) return '';
    return noteContent.includes('<') ? stripHtml(noteContent) : noteContent;
  }, [noteContent]);

  const topics = useMemo(() => extractTopics(plainText), [plainText]);

  const hasContent = plainText.length >= 20;

  useEffect(() => {
    let active = true;
    const check = () => {
      fetch('/api/ai/status')
        .then(r => r.json())
        .then(data => {
          if (!active) return;
          const connected = data.llm?.available ?? data.llm_server?.connected ?? false;
          setAiAvailable(connected);
        })
        .catch(() => { if (active) setAiAvailable(false); });
    };
    check();
    const id = setInterval(check, aiAvailable ? 30000 : 3000);
    return () => { active = false; clearInterval(id); };
  }, [aiAvailable]);

  useEffect(() => {
    const prevId = prevNoteIdRef.current;
    if (prevId !== noteId) {
      if (prevId) {
        saveCurrentState(prevId);
      }

      abortRef.current.abort();
      abortRef.current = new AbortController();

      if (noteId) {
        const cached = getDocState(noteId);
        loadState(cached);
      } else {
        loadState(defaultDocState());
      }

      noteIdRef.current = noteId;
      prevNoteIdRef.current = noteId;
    }
  }, [noteId, saveCurrentState, getDocState, loadState]);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages, chatStreaming]);

  const startResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizingRef.current = true;
    setIsResizing(true);
    resizeStartRef.current = { x: e.clientX, width: panelWidth };

    const onMouseMove = (ev: MouseEvent) => {
      if (!resizingRef.current) return;
      const diff = resizeStartRef.current.x - ev.clientX;
      const newWidth = Math.max(280, Math.min(600, resizeStartRef.current.width + diff));
      setPanelWidth(newWidth);
    };

    const onMouseUp = () => {
      resizingRef.current = false;
      setIsResizing(false);
      document.body.classList.remove('resizing-col');
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };

    document.body.classList.add('resizing-col');
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [panelWidth]);

  const blocksRef = useRef<string[]>([]);
  const blockSummariesRef = useRef<string[]>([]);
  const batchStartRef = useRef(0);

  // Process 3 blocks per batch, then pause for user to continue
  const runBatch = useCallback(async (startIdx: number) => {
    const docId = noteIdRef.current;
    const signal = abortRef.current.signal;
    setSummaryPhase('streaming');
    setProgressMsg('');

    try {
      const res = await fetch('/api/ai/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'summarize-batch',
          blocks: blocksRef.current,
          startIdx,
          batchSize: 3,
          previousBlockSummaries: blockSummariesRef.current,
        }),
        signal,
      });
      if (noteIdRef.current !== docId || !res.ok) return;

      await readStream(res, () => {},
        (msg) => { if (noteIdRef.current === docId) setProgressMsg(msg); },
        undefined, signal, undefined,
        (event) => {
          if (noteIdRef.current !== docId) return;
          blockSummariesRef.current = [...blockSummariesRef.current, event.summary];
          setSectionSummaries([...blockSummariesRef.current]);
          setCurrentSectionIdx(event.blockIndex);
          setCurrentStreamText('');
          setProgressMsg('');
        },
        (event) => {
          if (noteIdRef.current !== docId) return;
          setCurrentStreamText(event.text);
          setCurrentSectionIdx(event.blockIndex);
          setProgressMsg('');
        },
      );

      if (noteIdRef.current !== docId) return;
      batchStartRef.current = startIdx + 3;

      if (batchStartRef.current >= blocksRef.current.length) {
        setSummaryPhase('merging');
        setCurrentStreamText('');
        const synthRes = await fetch('/api/ai/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'summarize-final', allBlockSummaries: blockSummariesRef.current }),
          signal,
        });
        if (noteIdRef.current !== docId || !synthRes.ok) return;
        let final = '';
        await readStream(synthRes,
          (t) => { if (noteIdRef.current === docId) { final = t; setCurrentStreamText(t); } },
          (msg) => { if (noteIdRef.current === docId) setProgressMsg(msg); },
          undefined, signal,
        );
        if (noteIdRef.current !== docId) return;
        setMergedSummary(final);
        setCurrentStreamText('');
        setSummaryPhase('merged');
      } else {
        setSummaryPhase('section-done');
        setProgressMsg('');
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      setSummaryPhase('all-done');
    }
  }, []);

  const handleContinue = useCallback(() => {
    runBatch(batchStartRef.current);
  }, [runBatch]);

  const runSummarize = useCallback(async (hardRefresh = false) => {
    if (!noteId || !hasContent) return;
    const docId = noteId;
    const signal = abortRef.current.signal;

    setActiveMode('summarize');
    setSummaryPhase('preparing');
    setSectionChunks([]);
    setSectionCacheIds([]);
    setSectionSummaries([]);
    setCurrentSectionIdx(-1);
    setCurrentStreamText('');
    setMergedSummary('');
    setShowAllPoints({});
    setCopiedField(null);
    setProgressMsg('');
    blocksRef.current = [];
    blockSummariesRef.current = [];
    batchStartRef.current = 0;

    const inputText = plainText.slice(0, 30000);

    try {
      if (hardRefresh) {
        await fetch('/api/ai/cache', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'clear', sourceId: noteId }),
          signal,
        });
        if (noteIdRef.current !== docId) return;
      }

      const res = await fetch('/api/ai/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'summarize', text: inputText, sourceId: noteId }),
        signal,
      });
      if (noteIdRef.current !== docId || !res.ok) {
        setSectionSummaries(['Failed to start summarization.']);
        setSummaryPhase('all-done');
        return;
      }

      let gotBlocks = false;
      let finalResult = '';

      const reader = res.body?.getReader();
      if (!reader) { setSummaryPhase('all-done'); return; }
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        if (signal?.aborted) { reader.cancel(); return; }
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ') || trimmed === 'data: [DONE]') continue;
          try {
            const parsed = JSON.parse(trimmed.slice(6));
            if (parsed.blocks) {
              blocksRef.current = parsed.blocks;
              gotBlocks = true;
            }
            if (parsed.text) {
              finalResult = parsed.text;
              setSummaryPhase('merging');
              setCurrentStreamText(parsed.text);
            }
            if (parsed.progress) setProgressMsg(parsed.progress);
          } catch {}
        }
      }

      if (noteIdRef.current !== docId) return;

      if (gotBlocks) {
        await runBatch(0);
      } else if (finalResult) {
        setMergedSummary(finalResult);
        setCurrentStreamText('');
        setSummaryPhase('merged');
      } else {
        setSummaryPhase('all-done');
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      if (noteIdRef.current !== docId) return;
      setSectionSummaries(['Failed to generate summary. Please try again.']);
      setSummaryPhase('all-done');
    }
  }, [noteId, hasContent, plainText, runBatch]);

  const explainChunksRef = useRef<string[]>([]);
  const explainIdxRef = useRef(0);
  const explainResultsRef = useRef<string[]>([]);

  const explainOneChunk = useCallback(async (idx: number) => {
    const docId = noteIdRef.current;
    const signal = abortRef.current.signal;
    setExplainPhase('streaming');
    setExplainStreamText('');
    setProgressMsg('');

    try {
      const res = await fetch('/api/ai/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'explain-chunk',
          chunkText: explainChunksRef.current[idx],
          chunkIndex: idx,
          totalChunks: explainChunksRef.current.length,
        }),
        signal,
      });
      if (noteIdRef.current !== docId || !res.ok) return;

      let result = '';
      await readStream(res,
        (t) => { if (noteIdRef.current === docId) { result = t; setExplainStreamText(t); } },
        (msg) => { if (noteIdRef.current === docId) setProgressMsg(msg); },
        undefined, signal,
      );

      if (noteIdRef.current !== docId) return;

      explainResultsRef.current = [...explainResultsRef.current, result];
      setExplainSections(explainResultsRef.current.map(text => [{ text: '', explanation: text }]));
      setExplainCurrentIdx(idx);
      setExplainStreamText('');
      setExplainSource('llm');
      setProgressMsg('');

      explainIdxRef.current = idx + 1;
      if (explainIdxRef.current >= explainChunksRef.current.length) {
        setExplainPhase('all-done');
      } else {
        setExplainPhase('section-done');
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      setExplainPhase('all-done');
    }
  }, []);

  const handleExplainContinue = useCallback(() => {
    explainOneChunk(explainIdxRef.current);
  }, [explainOneChunk]);

  const runExplain = useCallback(async (hardRefresh = false) => {
    if (!noteId || !hasContent) return;
    const docId = noteId;
    const signal = abortRef.current.signal;

    setActiveMode('explain');
    setExplainPhase('preparing');
    setExplainChunks([]);
    setExplainCacheIds([]);
    setExplainSections([]);
    setExplainCurrentIdx(-1);
    setExplainStreamText('');
    setExplainSource('');
    setCopiedField(null);
    setProgressMsg('');
    explainChunksRef.current = [];
    explainIdxRef.current = 0;
    explainResultsRef.current = [];

    const inputText = plainText.slice(0, 30000);

    try {
      // Get 100-word chunks from backend
      const res = await fetch('/api/ai/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'explain', text: inputText, sourceId: noteId }),
        signal,
      });
      if (noteIdRef.current !== docId || !res.ok) {
        setExplainSections([[{ text: '', explanation: 'Failed to start explanation.' }]]);
        setExplainPhase('all-done');
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) { setExplainPhase('all-done'); return; }
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        if (signal?.aborted) { reader.cancel(); return; }
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ') || trimmed === 'data: [DONE]') continue;
          try {
            const parsed = JSON.parse(trimmed.slice(6));
            if (parsed.explainChunks) {
              explainChunksRef.current = parsed.explainChunks;
              setExplainChunks(parsed.explainChunks);
            }
          } catch {}
        }
      }

      if (noteIdRef.current !== docId) return;

      if (explainChunksRef.current.length > 0) {
        await explainOneChunk(0);
      } else {
        setExplainPhase('all-done');
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      if (noteIdRef.current !== docId) return;
      setExplainSections([[{ text: '', explanation: 'Failed to prepare explanation.' }]]);
      setExplainPhase('all-done');
    }
  }, [noteId, hasContent, plainText, explainOneChunk]);

  const isGreeting = useCallback((msg: string): boolean => {
    const trimmed = msg.trim().toLowerCase();
    return /^(hi|hello|hey|good\s*(morning|afternoon|evening|day)|howdy|what'?s\s*up|yo|sup|hola|greetings|thanks?|thank\s+you|bye|goodbye|ok|okay|cool|nice|great)\b/.test(trimmed);
  }, []);

  const GREETING_REPLIES = useMemo(() => [
    "Hello! I'm your document assistant. Ask me anything about the current document.",
    "Hi there! I can help you understand, summarize, or explain this document. What would you like to know?",
    "Hey! Ask me a question about the content and I'll find the answer from the document.",
  ], []);

  const sendChatMessage = useCallback(async () => {
    if (!noteId || !chatInput.trim() || chatLoading) return;

    const docId = noteId;
    const signal = abortRef.current.signal;
    const userMessage = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    if (isGreeting(userMessage)) {
      const reply = GREETING_REPLIES[Math.floor(Math.random() * GREETING_REPLIES.length)];
      setChatLoading(true);
      setChatStreaming(true);
      setChatMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      let i = 0;
      const step = () => {
        if (noteIdRef.current !== docId) return;
        i += 1 + Math.floor(Math.random() * 2);
        const partial = reply.slice(0, i);
        setChatMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: partial };
          return updated;
        });
        if (i < reply.length) {
          setTimeout(step, 18 + Math.random() * 12);
        } else {
          if (noteIdRef.current !== docId) return;
          setChatLoading(false);
          setChatStreaming(false);
        }
      };
      setTimeout(step, 80);
      return;
    }

    setChatLoading(true);
    setChatStreaming(true);

    setChatMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const streamRes = await fetch('/api/ai/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'chat',
          text: plainText.slice(0, 6000),
          sourceId: noteId,
          chatMessage: userMessage,
          chatHistory: chatMessages.slice(-10),
        }),
        signal,
      });

      if (noteIdRef.current !== docId) return;

      if (!streamRes.ok) {
        const errData = await streamRes.json().catch(() => ({}));
        const errMsg = errData.error || `Error ${streamRes.status}`;
        if (noteIdRef.current !== docId) return;
        setChatMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: errMsg };
          return updated;
        });
        setChatLoading(false);
        setChatStreaming(false);
        return;
      }

      if (streamRes.headers.get('content-type')?.includes('text/event-stream')) {
        let chatStreamError = '';
        let retrievalSources: ChatSource[] | undefined;
        let retrievalMethod: ChatMessage['retrievalMethod'];

        const fullText = await readStream(
          streamRes,
          (t) => {
            if (noteIdRef.current !== docId) return;
            setChatMessages(prev => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: 'assistant', content: cleanChatText(t), sources: retrievalSources, retrievalMethod };
              return updated;
            });
          },
          undefined,
          (err) => { chatStreamError = err; },
          signal,
          (data) => {
            retrievalSources = data.sources;
            retrievalMethod = data.method;
          },
        );

        if (noteIdRef.current !== docId) return;

        if (chatStreamError) {
          setChatMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: 'assistant', content: chatStreamError };
            return updated;
          });
        } else if (fullText) {
          setChatMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: 'assistant', content: cleanChatText(fullText), sources: retrievalSources, retrievalMethod };
            return updated;
          });
        }
      } else {
        const res = await fetch('/api/ai', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'chat',
            text: plainText.slice(0, 6000),
            sourceId: noteId,
            chatMessage: userMessage,
            chatHistory: chatMessages.slice(-10),
          }),
          signal,
        });

        if (noteIdRef.current !== docId) return;

        const data = await res.json();
        const retrieval = data.retrieval;
        setChatMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: cleanChatText(data.response || 'No response.'),
            sources: retrieval?.sources,
            retrievalMethod: retrieval?.method,
          };
          return updated;
        });
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      if (noteIdRef.current !== docId) return;
      console.error('Chat error:', err);
      setChatMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: 'assistant', content: 'Failed to get response. Please refresh.' };
        return updated;
      });
    }
    if (noteIdRef.current !== docId) return;
    setChatLoading(false);
    setChatStreaming(false);
  }, [noteId, chatInput, chatLoading, plainText, chatMessages, isGreeting, GREETING_REPLIES]);

  const handleChatKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  }, [sendChatMessage]);

  const copyText = useCallback((text: string, field: CopiedField) => {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopiedField(field);
      setTimeout(() => setCopiedField(null), 2000);
    }).catch(() => {});
  }, []);

  const copySummary = useCallback(() => {
    const text = mergedSummary || sectionSummaries.join('\n\n');
    copyText(text, 'summary');
  }, [mergedSummary, sectionSummaries, copyText]);

  const copyExplain = useCallback(() => {
    const allSentences = explainSections.flat();
    const text = allSentences.map(s => `${s.text}\n${s.explanation}`).join('\n\n');
    copyText(text, 'explain');
  }, [explainSections, copyText]);

  const handleTopicClick = useCallback((topic: string) => {
    setActiveTopic(topic);
    onTopicClick?.(topic);
    setTimeout(() => setActiveTopic(null), 3000);
  }, [onTopicClick]);

  const switchToSummarize = useCallback(() => {
    setActiveMode('summarize');
    const hasResults = sectionSummaries.length > 0 || mergedSummary;
    const isLoading = summaryPhase === 'preparing' || summaryPhase === 'streaming' || summaryPhase === 'merging';
    if (!hasResults && !isLoading) {
      runSummarize();
    }
  }, [summaryPhase, sectionSummaries.length, mergedSummary, runSummarize]);

  const switchToExplain = useCallback(() => {
    setActiveMode('explain');
    const hasResults = explainSections.length > 0;
    const isLoading = explainPhase === 'preparing' || explainPhase === 'streaming';
    if (!hasResults && !isLoading) {
      runExplain();
    }
  }, [explainPhase, explainSections.length, runExplain]);

  const switchToChat = useCallback(() => {
    setActiveMode('chat');
  }, []);

  if (isMobile && !noteId) return null;

  const isCollapsed = collapsed && !isMobile;
  const isAnyLoading = (summaryPhase === 'preparing' || summaryPhase === 'streaming' || summaryPhase === 'merging') || (explainPhase === 'preparing' || explainPhase === 'streaming') || chatLoading;

  return (
    <div className={isMobile ? 'flex h-full' : 'flex shrink-0 h-full'}>
      {!isMobile && !isCollapsed && (
        <div
          onMouseDown={startResize}
          className="shrink-0 w-px bg-border hover:bg-primary/40 cursor-col-resize transition-colors duration-150 relative group"
          title="Drag to resize"
        >
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-1/2 w-3 h-6 rounded-full bg-border group-hover:bg-primary/40 flex items-center justify-center transition-colors duration-150 z-10">
            <div className="w-[3px] h-3 rounded-full bg-muted-foreground/50 group-hover:bg-primary/70 transition-colors" />
          </div>
        </div>
      )}

      <aside
        className={`border-l border-border bg-card flex flex-col h-full overflow-hidden ${
          isMobile ? 'w-full' : (!isResizing ? 'transition-[width] duration-200 ease-out' : '')
        }`}
        style={isMobile ? undefined : { width: isCollapsed ? 48 : panelWidth }}
      >
        {isCollapsed ? (
          <div className="p-1.5 flex flex-col items-center">
            <Tooltip content="Expand insights" side="left">
              <button onClick={onToggleCollapse} className="w-8 h-8 rounded-md flex items-center justify-center hover:bg-accent transition-colors text-muted-foreground hover:text-foreground">
                <PanelRight size={16} />
              </button>
            </Tooltip>
          </div>
        ) : !noteId ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-2 px-6">
              <p className="text-xs text-muted-foreground/50">Select an item to analyze</p>
            </div>
          </div>
        ) : (
          <>
        <div className="p-3 space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold tracking-tight text-foreground">
              Insights
            </h3>
            <div className="flex items-center gap-0.5">
              {isMobile && onClose && (
                <button onClick={onClose} className="p-1 rounded-md hover:bg-accent text-muted-foreground text-xs">
                  Close
                </button>
              )}
              {!isMobile && onToggleCollapse && (
                <Tooltip content="Collapse insights" side="left">
                  <button onClick={onToggleCollapse} className="w-7 h-7 rounded-md flex items-center justify-center hover:bg-accent transition-colors text-muted-foreground hover:text-foreground">
                    <PanelRightClose size={14} />
                  </button>
                </Tooltip>
              )}
            </div>
          </div>

          {aiAvailable === false && (
            <div className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-muted/50">
              <div className="w-3 h-3 border-[1.5px] border-muted-foreground/30 border-t-foreground rounded-full animate-spin shrink-0" />
              <p className="text-[11px] text-muted-foreground">AI engine is loading...</p>
            </div>
          )}

          <div className="flex gap-0.5 p-0.5 bg-muted rounded-lg">
            <button
              onClick={switchToSummarize}
              disabled={isAnyLoading || !hasContent}
              className={`flex-1 flex items-center justify-center gap-1 text-[11px] py-1.5 rounded-md font-medium transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed ${
                activeMode === 'summarize'
                  ? 'bg-card text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {summaryPhase === 'preparing' && <LoadingDots />}
              Summary
            </button>
            <button
              onClick={switchToExplain}
              disabled={isAnyLoading || !hasContent}
              className={`flex-1 flex items-center justify-center gap-1 text-[11px] py-1.5 rounded-md font-medium transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed ${
                activeMode === 'explain'
                  ? 'bg-card text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {explainPhase === 'preparing' && <LoadingDots />}
              Explain
            </button>
            <button
              onClick={switchToChat}
              disabled={isAnyLoading || !hasContent}
              className={`flex-1 flex items-center justify-center gap-1 text-[11px] py-1.5 rounded-md font-medium transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed ${
                activeMode === 'chat'
                  ? 'bg-card text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Chat
            </button>
          </div>

          {!hasContent && (
            <p className="text-[10px] text-muted-foreground/50">Add more content to enable AI features</p>
          )}
        </div>

        <Separator />

        {activeMode === 'chat' ? (
          <div className="flex-1 flex flex-col min-h-0">
            <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-3 space-y-3">
              {chatMessages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center px-4">
                  <p className="text-[11px] text-muted-foreground/50">
                    Ask anything about this content
                  </p>
                </div>
              )}

              {chatMessages.map((msg, i) => (
                <div key={i} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`max-w-[85%] rounded-lg px-3 py-2 ${
                    msg.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted text-foreground'
                  }`}>
                    {msg.role === 'assistant' && !msg.content && chatLoading && i === chatMessages.length - 1 ? (
                      <TypingIndicator compact />
                    ) : (
                      <p className="text-[12px] leading-relaxed whitespace-pre-wrap">
                        {msg.content}
                        {chatStreaming && i === chatMessages.length - 1 && msg.role === 'assistant' && msg.content && (
                          <span className="inline-block w-1.5 h-3.5 bg-current/60 animate-pulse ml-0.5 align-middle" />
                        )}
                      </p>
                    )}
                  </div>
                  {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && msg.content && !chatStreaming && (
                    <div className="max-w-[85%] mt-1 space-y-1">
                      <p className="text-[9px] text-muted-foreground/60 uppercase tracking-wider font-medium px-1">
                        {msg.retrievalMethod === 'contextual' ? 'Matched sections' : 'Relevant passages'}
                      </p>
                      {msg.sources.map((src, si) => (
                        <button
                          key={si}
                          onClick={() => {
                            if (src.matchedTerm && onTopicClick) {
                              onTopicClick(src.matchedTerm);
                            }
                          }}
                          className="block w-full text-left rounded border border-border/50 bg-background/50 px-2 py-1.5 hover:bg-accent/50 transition-colors group"
                        >
                          <p className="text-[10px] text-muted-foreground leading-snug line-clamp-2">
                            {src.snippet}
                          </p>
                          {src.matchedTerm && (
                            <p className="text-[9px] text-primary/70 mt-0.5 group-hover:text-primary">
                              ~ word {src.wordOffset >= 0 ? src.wordOffset : '?'} &middot; &ldquo;{src.matchedTerm}&rdquo;
                            </p>
                          )}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="border-t border-border p-2.5">
              <div className="flex gap-2 items-end">
                <textarea
                  ref={chatInputRef}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={handleChatKeyDown}
                  placeholder="Ask about this content..."
                  rows={1}
                  className="flex-1 resize-none rounded-lg border border-input bg-transparent px-3 py-2 text-[12px] placeholder:text-muted-foreground/40 focus:outline-none focus:ring-1 focus:ring-ring min-h-[36px] max-h-[100px]"
                  style={{ height: 'auto' }}
                  onInput={(e) => {
                    const el = e.currentTarget;
                    el.style.height = 'auto';
                    el.style.height = Math.min(el.scrollHeight, 100) + 'px';
                  }}
                />
                <button
                  onClick={sendChatMessage}
                  disabled={!chatInput.trim() || chatLoading}
                  className="shrink-0 w-9 h-9 rounded-lg bg-primary text-primary-foreground flex items-center justify-center hover:bg-primary/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Send size={14} />
                </button>
              </div>
            </div>
          </div>
        ) : (
          <ScrollArea className="flex-1">
            <div className="p-1.5 space-y-0.5">

              {topics.length > 0 && (
                <div className="px-2 py-2.5">
                  <div className="mb-2.5">
                    <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Key Topics</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {topics.map((topic) => (
                      <button
                        key={topic}
                        onClick={() => handleTopicClick(topic)}
                        className="group"
                      >
                        <Badge
                          variant={activeTopic === topic ? 'default' : 'secondary'}
                          className={`text-[10px] px-2.5 py-1 rounded-full cursor-pointer transition-all duration-150 ${
                            activeTopic === topic
                              ? 'bg-primary/20 text-primary border-primary/30 shadow-sm'
                              : 'hover:bg-accent hover:text-foreground hover:border-muted-foreground/30'
                          }`}
                        >
                          {topic}
                        </Badge>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {topics.length > 0 && activeMode && (
                <Separator className="my-1" />
              )}

              {activeMode === 'summarize' && (
                <div className="px-2 py-2.5">
                  <div className="flex items-center justify-between mb-2.5">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Summary</span>
                      {sectionChunks.length > 1 && summaryPhase !== 'idle' && summaryPhase !== 'preparing' && (
                        <span className="text-[8px] font-mono px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">
                          {Math.min(currentSectionIdx + 1, sectionChunks.length)}/{sectionChunks.length}
                        </span>
                      )}
                      {(sectionSummaries.length > 0 || mergedSummary) && summaryPhase !== 'idle' && summaryPhase !== 'preparing' && (
                        <span className="text-[8px] font-mono px-1.5 py-0.5 rounded-full bg-green-500/15 text-green-500">
                          AI
                        </span>
                      )}
                    </div>
                    {(sectionSummaries.length > 0 || mergedSummary) && summaryPhase !== 'preparing' && summaryPhase !== 'streaming' && summaryPhase !== 'merging' && (
                      <div className="flex items-center gap-0.5">
                        <button
                          onClick={copySummary}
                          className={`w-7 h-7 rounded-md flex items-center justify-center transition-colors ${
                            copiedField === 'summary'
                              ? 'text-green-500 bg-green-500/10'
                              : 'text-muted-foreground hover:bg-accent hover:text-foreground'
                          }`}
                          title={copiedField === 'summary' ? 'Copied!' : 'Copy'}
                        >
                          {copiedField === 'summary' ? <Check size={13} /> : <Copy size={13} />}
                        </button>
                        <button
                          onClick={() => runSummarize(true)}
                          disabled={isAnyLoading}
                          className="w-7 h-7 rounded-md flex items-center justify-center text-muted-foreground hover:bg-accent hover:text-foreground transition-colors disabled:opacity-40"
                          title="Re-generate (hard refresh)"
                        >
                          <RefreshCw size={13} />
                        </button>
                      </div>
                    )}
                  </div>

                  {(summaryPhase === 'preparing' || (summaryPhase === 'streaming' && sectionSummaries.length === 0 && !currentStreamText)) && (
                    <SkeletonBlock lines={3} message={progressMsg || undefined} />
                  )}

                  {(sectionSummaries.length > 0 || (summaryPhase === 'streaming' && currentStreamText)) && summaryPhase !== 'merged' && summaryPhase !== 'merging' && (
                    <div className="space-y-1.5">
                      {sectionSummaries.map((section, i) => (
                        <div key={i} className="rounded-md border border-border/60 overflow-hidden">
                          <div className="flex items-center gap-1.5 px-3 py-1 bg-muted/30 border-b border-border/40">
                            <div className="w-1 h-1 rounded-full bg-primary/50" />
                            <span className="text-[9px] font-medium text-muted-foreground/70 uppercase tracking-widest">Section {i + 1}</span>
                          </div>
                          <div className="px-3 py-2">
                            <p className="text-[12px] text-foreground/80 leading-relaxed">{section}</p>
                          </div>
                        </div>
                      ))}

                      {summaryPhase === 'streaming' && currentStreamText && (
                        <div className="rounded-md border border-border/60 overflow-hidden">
                          <div className="flex items-center gap-1.5 px-3 py-1 bg-muted/30 border-b border-border/40">
                            <div className="w-1 h-1 rounded-full bg-primary/50" />
                            <span className="text-[9px] font-medium text-muted-foreground/70 uppercase tracking-widest">Section {currentSectionIdx + 1}</span>
                          </div>
                          <div className="px-3 py-2">
                            <p className="text-[12px] text-foreground/80 leading-relaxed">
                              {currentStreamText}
                              <span className="inline-block w-[2px] h-3.5 bg-primary/60 animate-pulse ml-0.5 align-middle rounded-full" />
                            </p>
                          </div>
                        </div>
                      )}

                      {summaryPhase === 'streaming' && !currentStreamText && progressMsg && (
                        <SkeletonBlock lines={2} message={progressMsg} />
                      )}

                      {summaryPhase === 'section-done' && (
                        <button
                          onClick={handleContinue}
                          className="w-full flex items-center justify-center py-2.5 rounded-md text-[11px] font-medium text-muted-foreground hover:text-foreground bg-muted/20 hover:bg-muted/40 border border-border/40 transition-all duration-200"
                        >
                          Continue
                        </button>
                      )}
                    </div>
                  )}

                  {summaryPhase === 'merging' && currentStreamText && (
                    <div className="rounded-md bg-muted/20 p-3 border border-border/30">
                      <p className="text-[12px] text-foreground/85 leading-[1.7] whitespace-pre-wrap">
                        {currentStreamText}
                        <span className="inline-block w-[2px] h-3.5 bg-primary/60 animate-pulse ml-0.5 align-middle rounded-full" />
                      </p>
                    </div>
                  )}

                  {summaryPhase === 'merging' && !currentStreamText && (
                    <SkeletonBlock lines={3} message="Writing final summary..." />
                  )}

                  {summaryPhase === 'merged' && mergedSummary && (
                    <StructuredSummaryView
                      text={mergedSummary}
                      showAllPoints={showAllPoints}
                      onToggleShowAll={(key) => setShowAllPoints(prev => ({ ...prev, [key]: !prev[key] }))}
                    />
                  )}

                  {summaryPhase === 'all-done' && sectionSummaries.length > 0 && !mergedSummary && (
                    <StructuredSummaryView
                      text={sectionSummaries.join('\n\n')}
                      showAllPoints={showAllPoints}
                      onToggleShowAll={(key) => setShowAllPoints(prev => ({ ...prev, [key]: !prev[key] }))}
                    />
                  )}

                  {summaryPhase === 'idle' && sectionSummaries.length === 0 && !mergedSummary && (
                    <div className="flex flex-col items-center justify-center py-6 text-center space-y-2.5">
                      <p className="text-[11px] text-muted-foreground/50">No summary yet</p>
                      <button
                        onClick={() => runSummarize()}
                        disabled={isAnyLoading || !hasContent}
                        className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-[11px] font-medium text-foreground bg-muted/40 hover:bg-accent border border-border/40 transition-colors disabled:opacity-40"
                      >
                        <RefreshCw size={12} />
                        Generate Summary
                      </button>
                    </div>
                  )}
                </div>
              )}

              {activeMode === 'explain' && (
                <div className="px-2 py-2.5">
                  <div className="flex items-center justify-between mb-2.5">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Explanation</span>
                      {explainChunks.length > 1 && explainPhase !== 'idle' && explainPhase !== 'preparing' && (
                        <span className="text-[8px] font-mono px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">
                          {Math.min(explainCurrentIdx + 1, explainChunks.length)}/{explainChunks.length}
                        </span>
                      )}
                      {explainSource && (
                        <span className={`text-[8px] font-mono px-1.5 py-0.5 rounded-full ${
                          explainSource === 'llm' ? 'bg-green-500/15 text-green-500' : 'bg-yellow-500/15 text-yellow-500'
                        }`}>
                          {explainSource === 'llm' ? 'AI' : 'basic'}
                        </span>
                      )}
                    </div>
                    {explainSections.length > 0 && explainPhase !== 'preparing' && explainPhase !== 'streaming' && (
                      <div className="flex items-center gap-0.5">
                        <button
                          onClick={copyExplain}
                          className={`w-7 h-7 rounded-md flex items-center justify-center transition-colors ${
                            copiedField === 'explain'
                              ? 'text-green-500 bg-green-500/10'
                              : 'text-muted-foreground hover:bg-accent hover:text-foreground'
                          }`}
                          title={copiedField === 'explain' ? 'Copied!' : 'Copy'}
                        >
                          {copiedField === 'explain' ? <Check size={13} /> : <Copy size={13} />}
                        </button>
                        <button
                          onClick={() => runExplain(true)}
                          disabled={isAnyLoading}
                          className="w-7 h-7 rounded-md flex items-center justify-center text-muted-foreground hover:bg-accent hover:text-foreground transition-colors disabled:opacity-40"
                          title="Re-generate (hard refresh)"
                        >
                          <RefreshCw size={13} />
                        </button>
                      </div>
                    )}
                  </div>

                  {(explainPhase === 'preparing' || (explainPhase === 'streaming' && explainSections.length === 0 && !explainStreamText)) && (
                    <SkeletonBlock lines={3} message={progressMsg || undefined} />
                  )}

                  {explainSections.length > 0 && (
                    <div className="space-y-1.5">
                      {explainSections.flat().filter(item => item.explanation).map((item, i) => (
                        <div key={i} className="rounded-md border border-border/60 overflow-hidden">
                          <div className="flex items-center gap-1.5 px-3 py-1 bg-muted/30 border-b border-border/40">
                            <div className="w-1 h-1 rounded-full bg-primary/50" />
                            <span className="text-[9px] font-medium text-muted-foreground/70 uppercase tracking-widest">Section {i + 1}</span>
                          </div>
                          <div className="px-3 py-2">
                            <p className="text-[12px] text-foreground/80 leading-relaxed">{item.explanation}</p>
                          </div>
                        </div>
                      ))}

                      {explainPhase === 'streaming' && explainStreamText && (
                        <div className="rounded-md border border-border/60 overflow-hidden">
                          <div className="flex items-center gap-1.5 px-3 py-1 bg-muted/30 border-b border-border/40">
                            <div className="w-1 h-1 rounded-full bg-primary/50" />
                            <span className="text-[9px] font-medium text-muted-foreground/70 uppercase tracking-widest">Section {explainCurrentIdx + 1}</span>
                          </div>
                          <div className="px-3 py-2">
                            <p className="text-[12px] text-foreground/80 leading-relaxed">
                              {explainStreamText}
                              <span className="inline-block w-[2px] h-3.5 bg-primary/60 animate-pulse ml-0.5 align-middle rounded-full" />
                            </p>
                          </div>
                        </div>
                      )}

                      {explainPhase === 'streaming' && !explainStreamText && progressMsg && (
                        <SkeletonBlock lines={2} message={progressMsg} />
                      )}

                      {explainPhase === 'section-done' && (
                        <button
                          onClick={handleExplainContinue}
                          className="w-full flex items-center justify-center py-2.5 rounded-md text-[11px] font-medium text-muted-foreground hover:text-foreground bg-muted/20 hover:bg-muted/40 border border-border/40 transition-all duration-200"
                        >
                          Continue
                        </button>
                      )}
                    </div>
                  )}

                  {explainSections.length === 0 && explainPhase === 'streaming' && explainStreamText && (
                    <div className="rounded-md bg-muted/20 p-3 border border-border/30">
                      <p className="text-[12px] text-foreground/85 leading-[1.7] whitespace-pre-wrap">
                        {explainStreamText}
                        <span className="inline-block w-[2px] h-3.5 bg-primary/60 animate-pulse ml-0.5 align-middle rounded-full" />
                      </p>
                    </div>
                  )}

                  {explainPhase === 'idle' && explainSections.length === 0 && !explainStreamText && (
                    <div className="flex flex-col items-center justify-center py-6 text-center space-y-2.5">
                      <p className="text-[11px] text-muted-foreground/50">No explanation yet</p>
                      <button
                        onClick={() => runExplain()}
                        disabled={isAnyLoading || !hasContent}
                        className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-[11px] font-medium text-foreground bg-muted/40 hover:bg-accent border border-border/40 transition-colors disabled:opacity-40"
                      >
                        <RefreshCw size={12} />
                        Generate Explanation
                      </button>
                    </div>
                  )}
                </div>
              )}

              {!activeMode && (
                <div className="flex flex-col items-center justify-center py-8 text-center space-y-2 px-4">
                  <p className="text-[10px] text-muted-foreground/40">
                    Click Summary, Explain, or Chat above
                  </p>
                </div>
              )}
            </div>
          </ScrollArea>
        )}
          </>
        )}
      </aside>
    </div>
  );
}

function LoadingDots() {
  return (
    <div className="flex gap-0.5">
      <div className="w-1 h-1 rounded-full bg-current animate-bounce" style={{ animationDelay: '0ms' }} />
      <div className="w-1 h-1 rounded-full bg-current animate-bounce" style={{ animationDelay: '150ms' }} />
      <div className="w-1 h-1 rounded-full bg-current animate-bounce" style={{ animationDelay: '300ms' }} />
    </div>
  );
}

function TypingIndicator({ compact }: { compact?: boolean }) {
  const barClass = "h-2.5 rounded-full animate-shimmer bg-[length:200%_100%] bg-gradient-to-r from-muted-foreground/10 via-muted-foreground/25 to-muted-foreground/10";
  if (compact) {
    return (
      <div className="py-0.5">
        <div className={`${barClass} w-24`} />
      </div>
    );
  }
  return (
    <div className="space-y-2.5 py-0.5">
      <div className={`${barClass} w-[90%]`} />
      <div className={`${barClass} w-[75%]`} />
      <div className={`${barClass} w-[60%]`} />
    </div>
  );
}

function StructuredSummaryView({ text, showAllPoints, onToggleShowAll, streaming }: {
  text: string;
  showAllPoints: Record<string, boolean>;
  onToggleShowAll: (key: string) => void;
  streaming?: boolean;
}) {
  const categories = parseStructuredSummary(text);
  const INITIAL_SHOW = 4;

  if (categories.length === 0) {
    return (
      <p className="text-[12px] text-foreground/85 leading-relaxed whitespace-pre-wrap">
        {text}
        {streaming && <span className="inline-block w-1.5 h-4 bg-primary/60 animate-pulse ml-0.5 align-middle" />}
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {categories.map((cat, i) => {
        const key = `${cat.title}-${i}`;
        const expanded = showAllPoints[key];
        const visiblePoints = expanded ? cat.points : cat.points.slice(0, INITIAL_SHOW);
        const hasMore = cat.points.length > INITIAL_SHOW;

        return (
          <div key={key}>
            <span className="text-[10px] font-semibold text-primary/70 uppercase tracking-wider">{cat.title}</span>
            <ul className="mt-1 space-y-1">
              {visiblePoints.map((point, j) => (
                <li key={j} className="flex gap-2 text-[12px] text-foreground/85 leading-relaxed">
                  <span className="text-primary/50 mt-0.5 shrink-0">-</span>
                  <span>{point}</span>
                </li>
              ))}
            </ul>
            {hasMore && !expanded && (
              <button
                onClick={() => onToggleShowAll(key)}
                className="mt-1 text-[10px] font-medium text-primary hover:text-primary/80 transition-colors"
              >
                Show {cat.points.length - INITIAL_SHOW} more →
              </button>
            )}
          </div>
        );
      })}
      {streaming && <span className="inline-block w-1.5 h-4 bg-primary/60 animate-pulse ml-0.5 align-middle" />}
    </div>
  );
}
