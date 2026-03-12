'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Dialog, DialogContent, Badge } from '@/components/ui/primitives';
import { Search, FileText, File, Image as ImageIcon } from 'lucide-react';
import type { Note, SearchResult } from '@/lib/types';
import { truncate } from '@/lib/utils';

interface SearchDialogProps {
  open: boolean;
  onClose: () => void;
  onSelect: (note: Note) => void;
  onViewDocument?: (id: string) => void;
  onViewImage?: (id: string) => void;
}

export default function SearchDialog({ open, onClose, onSelect, onViewDocument, onViewImage }: SearchDialogProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 50);
      setQuery('');
      setResults([]);
      setSelectedIdx(0);
    }
  }, [open]);

  const doSearch = useCallback(async (q: string) => {
    if (!q.trim()) { setResults([]); return; }
    setLoading(true);
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, topK: 10 }),
      });
      const data = await res.json();
      setResults(data);
      setSelectedIdx(0);
    } catch (err) {
      console.error('Search error:', err);
    }
    setLoading(false);
  }, []);

  const handleInput = (value: string) => {
    setQuery(value);
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => doSearch(value), 300);
  };

  const handleSelect = (result: SearchResult) => {
    const sourceType = result.source_type || 'note';
    if (sourceType === 'document' && onViewDocument) {
      onViewDocument(result.note.id);
    } else if (sourceType === 'image' && onViewImage) {
      onViewImage(result.note.id);
    } else {
      onSelect(result.note);
    }
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIdx(i => Math.min(i + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIdx(i => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && results[selectedIdx]) {
      handleSelect(results[selectedIdx]);
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  const sourceIcon = (type?: string) => {
    if (type === 'document') return <File size={14} className="mt-0.5 text-red-400 shrink-0" />;
    if (type === 'image') return <ImageIcon size={14} className="mt-0.5 text-green-400 shrink-0" />;
    return <FileText size={14} className="mt-0.5 text-muted-foreground shrink-0" />;
  };

  const sourceLabel = (type?: string) => {
    if (type === 'document') return 'DOC';
    if (type === 'image') return 'IMG';
    return null;
  };

  return (
    <Dialog open={open} onClose={onClose}>
      <DialogContent>
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Search size={16} className="text-muted-foreground shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => handleInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search notes, documents, images..."
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          {loading && (
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin shrink-0" />
          )}
          <kbd className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded shrink-0">ESC</kbd>
        </div>

        <div className="max-h-80 overflow-y-auto">
          {results.length === 0 && query && !loading ? (
            <div className="px-4 py-8 text-center">
              <p className="text-sm text-muted-foreground">No results found</p>
              <p className="text-xs text-muted-foreground/60 mt-1">Try different keywords</p>
            </div>
          ) : results.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <Search size={20} className="mx-auto text-muted-foreground/40 mb-2" />
              <p className="text-xs text-muted-foreground/60">Search across notes, documents, and images</p>
            </div>
          ) : (
            <div className="p-1.5">
              {results.map((result, idx) => (
                <button
                  key={`${result.note.id}-${idx}`}
                  onClick={() => handleSelect(result)}
                  onMouseEnter={() => setSelectedIdx(idx)}
                  className={`w-full text-left flex items-start gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                    idx === selectedIdx ? 'bg-accent' : 'hover:bg-accent/50'
                  }`}
                >
                  {sourceIcon(result.source_type)}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium truncate">{result.note.title}</span>
                      {sourceLabel(result.source_type) && (
                        <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground shrink-0">
                          {sourceLabel(result.source_type)}
                        </span>
                      )}
                      <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded-full shrink-0 ${
                        result.score > 0.6 ? 'bg-green-500/15 text-green-400' :
                        result.score > 0.3 ? 'bg-yellow-500/15 text-yellow-400' :
                        'bg-muted text-muted-foreground'
                      }`}>
                        {Math.round(result.score * 100)}%
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                      {truncate(result.chunk.content, 150)}
                    </p>
                    {result.matchedTopics && result.matchedTopics.length > 0 && (
                      <div className="flex gap-1 mt-1">
                        {result.matchedTopics.slice(0, 3).map(t => (
                          <Badge key={t.id} variant="outline" className="text-[8px]">{t.name}</Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between px-4 py-2 border-t border-border text-[10px] text-muted-foreground">
          <span>
            <kbd className="bg-muted px-1 rounded">↑↓</kbd> Navigate
            <kbd className="bg-muted px-1 rounded ml-2">↵</kbd> Open
          </span>
          <span>{results.length} results</span>
        </div>
      </DialogContent>
    </Dialog>
  );
}
