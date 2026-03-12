'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  FileText, Image as ImageIcon, X, Loader2, Clock, CalendarPlus,
  ZoomIn, ZoomOut, Maximize2, Copy, Check, Search,
  ChevronUp, ChevronDown,
} from 'lucide-react';
import type { Document, UploadedImage } from '@/lib/types';

interface DocumentViewerProps {
  type: 'document' | 'image';
  id: string;
  onClose: () => void;
  isMobile?: boolean;
  onTextChange?: (id: string, type: 'document' | 'image', text: string) => void;
}

type ViewerData = (Document | UploadedImage) & { chunks?: { id: string; content: string }[] };

const INITIAL_RENDER = 200;
const BATCH_SIZE = 200;
const SEARCH_DEBOUNCE = 200;

interface SearchMatch {
  paraIdx: number;
  startInPara: number;
  length: number;
}

function findMatches(paragraphs: string[], query: string): SearchMatch[] {
  if (!query) return [];
  const lower = query.toLowerCase();
  const matches: SearchMatch[] = [];
  for (let p = 0; p < paragraphs.length; p++) {
    const text = paragraphs[p].toLowerCase();
    let pos = text.indexOf(lower);
    while (pos !== -1) {
      matches.push({ paraIdx: p, startInPara: pos, length: lower.length });
      pos = text.indexOf(lower, pos + lower.length);
    }
  }
  return matches;
}

function renderHighlightedParagraph(
  text: string,
  matches: { start: number; length: number; active: boolean }[],
) {
  if (matches.length === 0) return text;

  const sorted = [...matches].sort((a, b) => a.start - b.start);
  const parts: React.ReactNode[] = [];
  let lastIdx = 0;

  for (let i = 0; i < sorted.length; i++) {
    const m = sorted[i];
    if (m.start > lastIdx) parts.push(text.slice(lastIdx, m.start));
    parts.push(
      <mark
        key={i}
        data-match-active={m.active ? 'true' : undefined}
        className={`rounded-sm px-0.5 ${m.active ? 'bg-yellow-400/80 text-black' : 'bg-yellow-400/30'}`}
      >
        {text.slice(m.start, m.start + m.length)}
      </mark>
    );
    lastIdx = m.start + m.length;
  }
  if (lastIdx < text.length) parts.push(text.slice(lastIdx));

  return <>{parts}</>;
}

export default function DocumentViewer({ type, id, onClose, isMobile, onTextChange }: DocumentViewerProps) {
  const [data, setData] = useState<ViewerData | null>(null);
  const [loading, setLoading] = useState(true);
  const [imageModal, setImageModal] = useState(false);
  const [imageZoom, setImageZoom] = useState(1);
  const [copied, setCopied] = useState(false);

  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [matchIndex, setMatchIndex] = useState(0);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  const [renderedCount, setRenderedCount] = useState(INITIAL_RENDER);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const copyText = useCallback((textToCopy: string) => {
    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    setSearchOpen(false);
    setSearchQuery('');
    setDebouncedQuery('');
    setRenderedCount(INITIAL_RENDER);
    const endpoint = type === 'document' ? `/api/documents/${id}` : `/api/images/${id}`;
    fetch(endpoint)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [type, id]);

  useEffect(() => {
    if (!data || data.status !== 'processing') return;
    const endpoint = type === 'document' ? `/api/documents/${id}` : `/api/images/${id}`;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(endpoint);
        const updated = await res.json();
        setData(updated);
        if (updated.status !== 'processing') clearInterval(interval);
      } catch {}
    }, 2000);
    const timeout = setTimeout(() => clearInterval(interval), 300000);
    return () => { clearInterval(interval); clearTimeout(timeout); };
  }, [data?.status, type, id]);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(searchQuery), SEARCH_DEBOUNCE);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const formatTimestamp = (ts: string) => {
    const d = new Date(ts + 'Z');
    if (isNaN(d.getTime())) return ts;
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  };

  const text = useMemo(() => {
    if (!data) return '';
    return type === 'document' ? (data as Document).plain_text : (data as UploadedImage).ocr_text;
  }, [data, type]);

  const paragraphs = useMemo(() => text ? text.split('\n\n') : [], [text]);

  useEffect(() => {
    setRenderedCount(INITIAL_RENDER);
  }, [text]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || renderedCount >= paragraphs.length) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setRenderedCount(prev => Math.min(prev + BATCH_SIZE, paragraphs.length));
        }
      },
      { rootMargin: '600px' },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [renderedCount, paragraphs.length]);

  const matches = useMemo(() => findMatches(paragraphs, debouncedQuery), [paragraphs, debouncedQuery]);

  useEffect(() => {
    if (matches.length > 0) setMatchIndex(0);
  }, [matches.length]);

  const matchesByPara = useMemo(() => {
    const map = new Map<number, { start: number; length: number; globalIdx: number }[]>();
    matches.forEach((m, i) => {
      if (!map.has(m.paraIdx)) map.set(m.paraIdx, []);
      map.get(m.paraIdx)!.push({ start: m.startInPara, length: m.length, globalIdx: i });
    });
    return map;
  }, [matches]);

  useEffect(() => {
    if (matches.length === 0) return;
    const active = matches[matchIndex];
    if (active && active.paraIdx >= renderedCount) {
      setRenderedCount(active.paraIdx + BATCH_SIZE);
    }
  }, [matchIndex, matches, renderedCount]);

  useEffect(() => {
    if (matches.length === 0) return;
    requestAnimationFrame(() => {
      const el = contentRef.current?.querySelector('[data-match-active="true"]');
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  }, [matchIndex, matches.length]);

  const nextMatch = () => setMatchIndex(i => (i + 1) % Math.max(1, matches.length));
  const prevMatch = () => setMatchIndex(i => (i - 1 + matches.length) % Math.max(1, matches.length));

  const toggleSearch = useCallback(() => {
    setSearchOpen(prev => {
      if (!prev) setTimeout(() => searchInputRef.current?.focus(), 50);
      else { setSearchQuery(''); setDebouncedQuery(''); }
      return !prev;
    });
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 size={24} className="animate-spin text-primary" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-muted-foreground">Content not found</p>
      </div>
    );
  }

  const isDoc = type === 'document';
  const title = isDoc ? (data as Document).filename : (data as UploadedImage).filename;
  const status = (data as Document | UploadedImage).status;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className={isMobile ? 'px-4 pt-3 pb-2 border-b border-border' : 'px-6 pt-5 pb-3 border-b border-border'}>
        <div className="flex items-center gap-3 mb-2">
          {isDoc ? <FileText size={isMobile ? 16 : 18} className="text-primary" /> : <ImageIcon size={isMobile ? 16 : 18} className="text-primary" />}
          <h2 className={`font-bold truncate flex-1 ${isMobile ? 'text-base' : 'text-xl'}`}>{title}</h2>
          {status === 'processing' && (
            <div className="flex items-center gap-1.5 text-xs text-yellow-400">
              <Loader2 size={12} className="animate-spin" /> Processing
            </div>
          )}
        </div>

        <div className={`flex items-center text-muted-foreground flex-wrap ${isMobile ? 'gap-2 text-[10px]' : 'gap-4 text-xs'}`}>
          {isDoc && <span>{(data as Document).page_count} pages</span>}
          {!isDoc && <span>{(data as UploadedImage).width}x{(data as UploadedImage).height}px</span>}
          <span>{(data.file_size / 1024).toFixed(1)} KB</span>
          {data.chunk_count > 0 && <span>{data.chunk_count} chunks</span>}
          <span className="flex items-center gap-1"><CalendarPlus size={isMobile ? 8 : 10} />Created: {formatTimestamp(data.created_at)}</span>
          <span className="flex items-center gap-1"><Clock size={isMobile ? 8 : 10} />Modified: {formatTimestamp(data.updated_at)}</span>
        </div>
      </div>

      {text && (
        <div className={`flex items-center justify-between py-2 border-b border-border ${isMobile ? 'px-4' : 'px-6'}`}>
          <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
            Extracted Text
          </span>
          <div className="flex items-center gap-0.5">
            <button
              onClick={toggleSearch}
              className={`flex items-center gap-1 px-2 py-1.5 rounded-md text-sm transition-colors shrink-0 ${
                searchOpen
                  ? 'text-primary bg-primary/10'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              }`}
              title="Search in text"
            >
              <Search size={14} />
            </button>
            <button
              onClick={() => copyText(text)}
              className={`flex items-center gap-1 px-2 py-1.5 rounded-md text-sm transition-colors shrink-0 ${
                copied
                  ? 'text-green-500 bg-green-500/10'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              }`}
              title={copied ? 'Copied!' : 'Copy text'}
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </div>
        </div>
      )}

      {searchOpen && (
        <div className={`flex items-center gap-2 py-2 border-b border-border bg-muted/30 ${isMobile ? 'px-4' : 'px-6'}`}>
          <Search size={13} className="text-muted-foreground shrink-0" />
          <input
            ref={searchInputRef}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.shiftKey ? prevMatch() : nextMatch(); }
              if (e.key === 'Escape') { setSearchOpen(false); setSearchQuery(''); setDebouncedQuery(''); }
            }}
            placeholder="Find in text..."
            className="flex-1 bg-transparent text-[12px] outline-none placeholder:text-muted-foreground/40"
          />
          {debouncedQuery && matches.length > 0 && (
            <span className="text-[10px] text-muted-foreground shrink-0">
              {matchIndex + 1}/{matches.length}
            </span>
          )}
          {debouncedQuery && matches.length === 0 && (
            <span className="text-[10px] text-muted-foreground/50 shrink-0">No matches</span>
          )}
          <button onClick={prevMatch} disabled={matches.length === 0} className="p-0.5 rounded hover:bg-accent text-muted-foreground disabled:opacity-30">
            <ChevronUp size={14} />
          </button>
          <button onClick={nextMatch} disabled={matches.length === 0} className="p-0.5 rounded hover:bg-accent text-muted-foreground disabled:opacity-30">
            <ChevronDown size={14} />
          </button>
          <button onClick={() => { setSearchOpen(false); setSearchQuery(''); setDebouncedQuery(''); }} className="p-0.5 rounded hover:bg-accent text-muted-foreground">
            <X size={14} />
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        {!isDoc && (data as UploadedImage).file_path && (
          <div className={isMobile ? 'px-4 pt-3' : 'px-6 pt-4'}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={(data as UploadedImage).file_path}
              alt={title}
              className="max-w-full rounded-lg border border-border cursor-pointer hover:opacity-90 transition-opacity"
              onClick={() => { setImageZoom(1); setImageModal(true); }}
            />
          </div>
        )}

        <div className={isMobile ? 'px-4 py-3' : 'px-6 py-4'}>
          {text ? (
            <div ref={contentRef} className="prose prose-invert max-w-none">
              {paragraphs.slice(0, renderedCount).map((para, i) => {
                const paraMatches = matchesByPara.get(i);
                return (
                  <p key={i} className="text-sm text-foreground/90 leading-relaxed mb-3 virtual-para">
                    {paraMatches
                      ? renderHighlightedParagraph(
                          para,
                          paraMatches.map(m => ({ start: m.start, length: m.length, active: m.globalIdx === matchIndex }))
                        )
                      : para
                    }
                  </p>
                );
              })}
              {renderedCount < paragraphs.length && (
                <div ref={sentinelRef} className="py-4 text-center">
                  <span className="text-xs text-muted-foreground/50">
                    Showing {renderedCount.toLocaleString()} of {paragraphs.length.toLocaleString()} paragraphs
                  </span>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground italic">
              {status === 'processing' ? 'Text extraction in progress...' : 'No text content extracted'}
            </p>
          )}
        </div>
      </div>

      {imageModal && !isDoc && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={() => setImageModal(false)}>
          <div className="relative w-[90vw] h-[85vh] bg-card rounded-xl border border-border shadow-2xl flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
              <span className="text-sm font-medium truncate flex items-center gap-2">
                <ImageIcon size={14} className="text-green-400" />
                {title}
              </span>
              <div className="flex items-center gap-1">
                <button onClick={() => setImageZoom(z => Math.max(0.25, z - 0.25))} className="p-1.5 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground transition-colors">
                  <ZoomOut size={16} />
                </button>
                <span className="text-xs text-muted-foreground w-12 text-center">{Math.round(imageZoom * 100)}%</span>
                <button onClick={() => setImageZoom(z => Math.min(4, z + 0.25))} className="p-1.5 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground transition-colors">
                  <ZoomIn size={16} />
                </button>
                <button onClick={() => setImageZoom(1)} className="p-1.5 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground transition-colors">
                  <Maximize2 size={16} />
                </button>
                <button onClick={() => setImageModal(false)} className="p-1.5 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground transition-colors ml-2">
                  <X size={16} />
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-auto flex items-center justify-center p-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={(data as UploadedImage).file_path}
                alt={title}
                className="transition-transform duration-200"
                style={{ transform: `scale(${imageZoom})`, transformOrigin: 'center' }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
