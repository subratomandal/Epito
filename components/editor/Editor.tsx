'use client';

import { useEditor, EditorContent } from '@tiptap/react';
import { Extension, Mark, mergeAttributes } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import TiptapLink from '@tiptap/extension-link';
import TaskList from '@tiptap/extension-task-list';
import TaskItem from '@tiptap/extension-task-item';
import Highlight from '@tiptap/extension-highlight';
import TiptapImage from '@tiptap/extension-image';
import TextAlign from '@tiptap/extension-text-align';
import CodeBlockLowlight from '@tiptap/extension-code-block-lowlight';
import { common, createLowlight } from 'lowlight';
import { Plugin, PluginKey } from '@tiptap/pm/state';
import { Decoration, DecorationSet } from '@tiptap/pm/view';
import type { Node as PMNode } from '@tiptap/pm/model';
import {
  useState, useEffect, useRef, useCallback, useMemo,
  forwardRef, useImperativeHandle, memo,
} from 'react';
import {
  Bold, Italic, Underline as UnderlineIcon, Strikethrough, Code,
  Heading1, Heading2, Heading3,
  List, ListOrdered, Quote, Minus, Highlighter, CheckSquare,
  FileText, Clock, CalendarPlus, Copy, Check, Search, X,
  ChevronUp, ChevronDown, AArrowUp, AArrowDown,
  Download, FileImage, FileType, Loader2,
  AlignLeft, AlignCenter, AlignRight,
} from 'lucide-react';
import { exportAsPDF, exportAsDOCX, exportAsImage, preloadExportLibs } from '@/inference/export';
import type { ExportFormat } from '@/inference/export';

const lowlight = createLowlight(common);

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.platform);
const MOD = IS_MAC ? '⌘' : 'Ctrl';
const ALT = IS_MAC ? '⌥' : 'Alt';

const PASTE_HTML_MAX = 200_000;
const PASTE_TEXT_MAX = 5_000_000;
const SEARCH_DEBOUNCE = 250;
const SEARCH_MAX_MATCHES = 500;

const FONT_SIZE_KEY = 'epito-editor-font-size';
const FONT_SIZE_DEFAULT = 15;
const FONT_SIZE_MIN = 10;
const FONT_SIZE_MAX = 28;
const FONT_SIZE_STEP = 1;

interface SearchState {
  query: string;
  activeIdx: number;
  set: DecorationSet;
  count: number;
}

const searchKey = new PluginKey<SearchState>('search');

function buildDecorations(
  doc: PMNode, query: string, activeIdx: number,
): { set: DecorationSet; count: number } {
  if (!query) return { set: DecorationSet.empty, count: 0 };
  const lower = query.toLowerCase();
  const decos: Decoration[] = [];
  let idx = 0;
  let capped = false;

  doc.descendants((node, pos) => {
    if (capped || !node.isText || !node.text) return;
    const text = node.text.toLowerCase();
    let found = text.indexOf(lower);
    while (found !== -1) {
      if (idx >= SEARCH_MAX_MATCHES) { capped = true; return; }
      decos.push(
        Decoration.inline(pos + found, pos + found + lower.length, {
          class: idx === activeIdx ? 'search-hl-active' : 'search-hl',
          'data-search-idx': String(idx),
        }),
      );
      idx++;
      found = text.indexOf(lower, found + lower.length);
    }
  });

  return { set: DecorationSet.create(doc, decos), count: idx };
}

const searchPlugin = new Plugin<SearchState>({
  key: searchKey,
  state: {
    init(): SearchState {
      return { query: '', activeIdx: 0, set: DecorationSet.empty, count: 0 };
    },
    apply(tr, prev): SearchState {
      const meta = tr.getMeta(searchKey) as
        | { query: string; activeIdx: number }
        | undefined;
      if (meta) {
        const { set, count } = buildDecorations(tr.doc, meta.query, meta.activeIdx);
        return { query: meta.query, activeIdx: meta.activeIdx, set, count };
      }
      if (tr.docChanged && prev.query) {
        const { set, count } = buildDecorations(tr.doc, prev.query, prev.activeIdx);
        return { ...prev, set, count };
      }
      return prev;
    },
  },
  props: {
    decorations(state) {
      return searchKey.getState(state)?.set ?? DecorationSet.empty;
    },
  },
});

const SearchExt = Extension.create({
  name: 'searchHighlight',
  addProseMirrorPlugins() {
    return [searchPlugin];
  },
});

const ClearMarksExt = Extension.create({
  name: 'clearMarksOnEmptyParagraph',
  addProseMirrorPlugins() {
    return [
      new Plugin({
        appendTransaction(transactions, _oldState, newState) {
          if (!transactions.some(tr => tr.docChanged)) return null;

          const { $from } = newState.selection;
          if (
            $from.parent.type.name === 'paragraph' &&
            $from.parent.content.size === 0 &&
            newState.storedMarks &&
            newState.storedMarks.length > 0
          ) {
            return newState.tr.setStoredMarks([]);
          }
          return null;
        },
      }),
    ];
  },
});

const UnderlineMark = Mark.create({
  name: 'underline',
  parseHTML() {
    return [
      { tag: 'u' },
      { style: 'text-decoration', getAttrs: (value) => (value as string).includes('underline') ? {} : false },
    ];
  },
  renderHTML({ HTMLAttributes }) {
    return ['u', mergeAttributes(HTMLAttributes), 0];
  },
  addKeyboardShortcuts() {
    return {
      'Mod-u': () => this.editor.commands.toggleMark(this.name),
    };
  },
});

const caretKey = new PluginKey('custom-caret');

const CustomCaretExt = Extension.create({
  name: 'customCaret',
  addProseMirrorPlugins() {
    return [
      new Plugin({
        key: caretKey,
        props: {
          decorations(state) {
            const { selection } = state;
            if (!selection.empty) return DecorationSet.empty;

            const widget = Decoration.widget(
              selection.head,
              () => {
                const el = document.createElement('span');
                el.className = 'custom-caret';
                return el;
              },
              { side: -1, key: `c-${selection.head}` },
            );

            return DecorationSet.create(state.doc, [widget]);
          },
        },
      }),
    ];
  },
});

const PlatformShortcutsExt = Extension.create({
  name: 'platformShortcuts',
  addKeyboardShortcuts() {
    const shortcuts: Record<string, () => boolean> = {
      'Mod-Alt-1': () => this.editor.commands.toggleHeading({ level: 1 }),
      'Mod-Alt-2': () => this.editor.commands.toggleHeading({ level: 2 }),
      'Mod-Alt-3': () => this.editor.commands.toggleHeading({ level: 3 }),
      'Mod-Shift-x': () => this.editor.commands.toggleStrike(),
    };

    if (!IS_MAC) {
      shortcuts['Ctrl-y'] = () => this.editor.commands.redo();
    }

    return shortcuts;
  },
});

function sanitizeExternalHTML(html: string): string {
  let h = html;

  h = h.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');
  h = h.replace(/<meta[^>]*\/?>/gi, '');
  h = h.replace(/<link[^>]*\/?>/gi, '');
  h = h.replace(/<!--[\s\S]*?-->/g, '');

  h = h.replace(/<\/?\w+:[^>]*>/g, '');

  h = h.replace(/\s*mso-[^;:"']+:[^;:"']+;?/gi, '');

  h = h.replace(/\s+class="[^"]*"/gi, '');

  h = h.replace(/<b\s+style="[^"]*font-weight:\s*normal[^"]*"\s*>/gi, '<span>');

  h = h.replace(/<span[^>]*>\s*<\/span>/gi, '');

  h = h.replace(/\s+data-[a-z-]+="[^"]*"/gi, '');
  h = h.replace(/\s+id="[^"]*"/gi, '');

  h = h.replace(/font-family:[^;}"']+;?/gi, '');
  h = h.replace(/line-height:[^;}"']+;?/gi, '');

  h = h.replace(/\s+style="\s*"/gi, '');

  return h;
}

function degradeLargePaste(html: string): string {
  if (html.length <= PASTE_HTML_MAX) return html;

  let text = html
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/(?:p|div|h[1-6]|li|tr|blockquote)>/gi, '\n\n')
    .replace(/<[^>]*>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'");

  if (text.length > PASTE_TEXT_MAX) text = text.slice(0, PASTE_TEXT_MAX);

  const safe = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  const paras = safe.split(/\n{2,}/).filter(p => p.trim());
  return paras.length > 0
    ? paras.map(p => `<p>${p.trim().replace(/\n/g, '<br>')}</p>`).join('')
    : '<p></p>';
}

function transformPastedHTMLPipeline(html: string): string {
  return degradeLargePaste(sanitizeExternalHTML(html));
}

function truncateLargePasteText(text: string): string {
  let t = text.length > PASTE_TEXT_MAX ? text.slice(0, PASTE_TEXT_MAX) : text;

  // Detect OCR-like text (mostly short lines) and rejoin into flowing paragraphs
  const lines = t.split('\n');
  if (lines.length > 3) {
    const shortLines = lines.filter(l => l.trim().length > 0 && l.trim().length < 60).length;
    const totalNonEmpty = lines.filter(l => l.trim().length > 0).length;
    if (totalNonEmpty > 0 && shortLines / totalNonEmpty > 0.6) {
      t = t
        .replace(/(\w)-\n(\w)/g, '$1$2')
        .replace(/\n{2,}/g, '\x00PARA\x00')
        .replace(/\n/g, ' ')
        .replace(/\x00PARA\x00/g, '\n\n')
        .replace(/ {2,}/g, ' ')
        .replace(/ *\n\n */g, '\n\n')
        .split('\n').map(l => l.trim()).join('\n')
        .trim();
    }
  }

  return t;
}

export interface NoteEditorHandle {
  getHTML: () => string;
  insertContent: (html: string) => void;
  replaceContent: (html: string) => void;
}

interface NoteEditorProps {
  noteId: string | null;
  content: string;
  title: string;
  createdAt?: string;
  updatedAt?: string;
  onTitleChange: (title: string) => void;
  onContentDirty: () => void;
  isMobile?: boolean;
}

const NoteEditor = forwardRef<NoteEditorHandle, NoteEditorProps>(
  function NoteEditor(
    { noteId, content, title, createdAt, updatedAt,
      onTitleChange, onContentDirty, isMobile },
    ref,
  ) {
    const loadedContentRef = useRef(content);
    const lastNoteIdRef = useRef<string | null>(noteId);
    const onContentDirtyRef = useRef(onContentDirty);
    const onTitleChangeRef = useRef(onTitleChange);

    const [copied, setCopied] = useState(false);
    const [exportOpen, setExportOpen] = useState(false);
    const [exporting, setExporting] = useState<ExportFormat | null>(null);
    const [exportError, setExportError] = useState<string | null>(null);

    const [searchOpen, setSearchOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [matchIndex, setMatchIndex] = useState(0);
    const [matchCount, setMatchCount] = useState(0);
    const searchInputRef = useRef<HTMLInputElement>(null);
    const searchTimerRef = useRef<ReturnType<typeof setTimeout>>();

    const [fontSize, setFontSize] = useState(() => {
      if (typeof window === 'undefined') return FONT_SIZE_DEFAULT;
      const saved = localStorage.getItem(FONT_SIZE_KEY);
      const parsed = saved ? parseInt(saved, 10) : NaN;
      return (parsed >= FONT_SIZE_MIN && parsed <= FONT_SIZE_MAX) ? parsed : FONT_SIZE_DEFAULT;
    });

    const changeFontSize = useCallback((delta: number) => {
      setFontSize(prev => {
        const next = Math.max(FONT_SIZE_MIN, Math.min(FONT_SIZE_MAX, prev + delta));
        localStorage.setItem(FONT_SIZE_KEY, String(next));
        return next;
      });
    }, []);

    const resetFontSize = useCallback(() => {
      setFontSize(FONT_SIZE_DEFAULT);
      localStorage.setItem(FONT_SIZE_KEY, String(FONT_SIZE_DEFAULT));
    }, []);

    useEffect(() => { onContentDirtyRef.current = onContentDirty; }, [onContentDirty]);
    useEffect(() => { onTitleChangeRef.current = onTitleChange; }, [onTitleChange]);

    useEffect(() => {
      setSearchOpen(false);
      setSearchQuery('');
      setMatchCount(0);
      setMatchIndex(0);
    }, [noteId]);

    const extensions = useMemo(() => [
      StarterKit.configure({ codeBlock: false }),
      CodeBlockLowlight.configure({ lowlight }),
      Placeholder.configure({ placeholder: 'Start writing...', showOnlyCurrent: false }),
      TiptapLink.configure({ openOnClick: false, HTMLAttributes: { target: '_blank', rel: 'noopener noreferrer' } }),
      TaskList,
      TaskItem.configure({ nested: true }),
      Highlight,
      TextAlign.configure({ types: ['heading', 'paragraph'] }),
      TiptapImage.configure({ inline: false, allowBase64: false }),
      UnderlineMark,
      SearchExt,
      ClearMarksExt,
      PlatformShortcutsExt,
      CustomCaretExt,
    ], []);

    const editor = useEditor({
      immediatelyRender: false,
      extensions,
      content,
      editorProps: {
        attributes: {
          class: 'tiptap focus:outline-none min-h-[400px] px-1',
        },
        transformPastedHTML: transformPastedHTMLPipeline,
        transformPastedText: truncateLargePasteText,
        handleClick: (_view, _pos, event) => {
          // Cmd/Ctrl+Click opens links (Notion pattern)
          if ((event.metaKey || event.ctrlKey) && event.target instanceof HTMLAnchorElement) {
            const href = event.target.getAttribute('href');
            if (href) {
              window.open(href, '_blank', 'noopener,noreferrer');
              return true;
            }
          }
          return false;
        },
      },
      onUpdate: () => {
        onContentDirtyRef.current();
      },
    });

    useImperativeHandle(ref, () => ({
      getHTML: () => editor?.getHTML() ?? loadedContentRef.current,
      insertContent: (html: string) => {
        if (!editor) return;
        editor.chain().focus().insertContent(html).run();
      },
      replaceContent: (html: string) => {
        if (!editor) return;
        editor.chain().focus().setContent(html).run();
      },
    }), [editor]);

    useEffect(() => {
      return () => clearTimeout(searchTimerRef.current);
    }, []);

    useEffect(() => {
      if (!editor) return;

      const isNoteSwitch = noteId !== lastNoteIdRef.current;
      lastNoteIdRef.current = noteId;

      if (isNoteSwitch) {
        editor.commands.setContent(content);
        loadedContentRef.current = content;
        return;
      }

      // Skip replacement if editor already has this content — avoids destroying
      // cursor position, undo history, and racing with user input on save roundtrips.
      if (content !== loadedContentRef.current) {
        if (editor.getHTML() === content) {
          loadedContentRef.current = content;
          return;
        }
        const { from, to } = editor.state.selection;
        editor.commands.setContent(content);
        loadedContentRef.current = content;
        try {
          const maxPos = editor.state.doc.content.size;
          editor.commands.setTextSelection({
            from: Math.min(from, maxPos),
            to: Math.min(to, maxPos),
          });
        } catch {}
      }
    }, [editor, content, noteId]);

    const copyPlainText = useCallback(() => {
      if (!editor) return;
      const { from, to, empty } = editor.state.selection;
      const text = empty
        ? editor.state.doc.textBetween(0, editor.state.doc.content.size, '\n', '\n')
        : editor.state.doc.textBetween(from, to, '\n', '\n');
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }).catch(() => {});
    }, [editor]);

    useEffect(() => {
      if (!exportOpen) return;
      const handleEsc = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
          setExportOpen(false);
          setExportError(null);
        }
      };
      window.addEventListener('keydown', handleEsc);
      return () => window.removeEventListener('keydown', handleEsc);
    }, [exportOpen]);

    const handleExport = useCallback(async (format: ExportFormat) => {
      if (!editor) return;

      // Close dialog immediately and show full-screen spinner
      setExportOpen(false);
      setExportError(null);
      setExporting(format);

      // Two rAF frames: first schedules paint, second guarantees it's flushed.
      // Without this, html2canvas blocks the main thread before the spinner renders.
      await new Promise<void>(r => {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => r());
        });
      });

      try {
        const html = editor.getHTML();
        const noteTitle = title || 'Untitled';
        if (format === 'pdf') await exportAsPDF(html, noteTitle);
        else if (format === 'docx') await exportAsDOCX(html, noteTitle);
        else if (format === 'png') await exportAsImage(html, noteTitle);
      } catch (err) {
        console.error(`Export as ${format} failed:`, err);
        const msg = err instanceof Error ? err.message : '';
        setExportOpen(true);
        setExportError(
          msg.includes('canvas') || msg.includes('memory') || msg.includes('size')
            ? 'Note is too large to export. Try splitting it into smaller notes.'
            : msg || `Failed to export as ${format.toUpperCase()}. Please try again.`
        );
      }
      setExporting(null);
    }, [editor, title]);

    const runSearch = useCallback((query: string, goToIdx?: number) => {
      if (!editor) return;
      const q = query.trim();
      const idx = goToIdx ?? 0;

      editor.view.dispatch(
        editor.state.tr.setMeta(searchKey, { query: q, activeIdx: idx }),
      );

      const ps = searchKey.getState(editor.state);
      const count = ps?.count ?? 0;
      setMatchCount(count);
      setMatchIndex(count > 0 ? ((idx % count) + count) % count : 0);

      if (count > 0) {
        requestAnimationFrame(() => {
          const el = editor.view.dom.querySelector(
            `[data-search-idx="${((idx % count) + count) % count}"]`,
          );
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
      }
    }, [editor]);

    useEffect(() => {
      if (!searchOpen) return;
      clearTimeout(searchTimerRef.current);
      if (!searchQuery.trim()) {
        if (editor) {
          editor.view.dispatch(
            editor.state.tr.setMeta(searchKey, { query: '', activeIdx: 0 }),
          );
        }
        setMatchCount(0);
        setMatchIndex(0);
        return;
      }
      searchTimerRef.current = setTimeout(() => runSearch(searchQuery), SEARCH_DEBOUNCE);
      return () => clearTimeout(searchTimerRef.current);
    }, [searchQuery, searchOpen, runSearch, editor]);

    const nextMatch = useCallback(() => {
      const next = (matchIndex + 1) % Math.max(1, matchCount);
      runSearch(searchQuery, next);
    }, [matchIndex, matchCount, searchQuery, runSearch]);

    const prevMatch = useCallback(() => {
      const prev = (matchIndex - 1 + matchCount) % Math.max(1, matchCount);
      runSearch(searchQuery, prev);
    }, [matchIndex, matchCount, searchQuery, runSearch]);

    const clearSearch = useCallback(() => {
      setSearchOpen(false);
      setSearchQuery('');
      if (editor) {
        editor.view.dispatch(
          editor.state.tr.setMeta(searchKey, { query: '', activeIdx: 0 }),
        );
      }
      setMatchCount(0);
      setMatchIndex(0);
    }, [editor]);

    const toggleSearch = useCallback(() => {
      setSearchOpen(prev => {
        if (!prev) {
          setTimeout(() => searchInputRef.current?.focus(), 50);
        } else {
          clearSearch();
        }
        return !prev;
      });
    }, [clearSearch]);

    useEffect(() => {
      if (!noteId) return;
      const handleKeyDown = (e: KeyboardEvent) => {
        const mod = e.metaKey || e.ctrlKey;
        if (!mod) return;

        if (e.key === 'f') {
          e.preventDefault();
          if (!searchOpen) {
            setSearchOpen(true);
            setTimeout(() => searchInputRef.current?.focus(), 50);
          } else {
            searchInputRef.current?.focus();
          }
          return;
        }

        if (e.key === '=' || e.key === '+') {
          e.preventDefault();
          changeFontSize(FONT_SIZE_STEP);
          return;
        }

        if (e.key === '-') {
          e.preventDefault();
          changeFontSize(-FONT_SIZE_STEP);
          return;
        }

        if (e.key === '0') {
          e.preventDefault();
          resetFontSize();
        }
      };
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }, [noteId, searchOpen, changeFontSize, resetFontSize]);

    const formatTimestamp = useCallback((ts?: string) => {
      if (!ts) return '';
      const d = new Date(ts + 'Z');
      if (isNaN(d.getTime())) return ts;
      return d.toLocaleString(undefined, {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    }, []);

    if (!noteId) {
      return (
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="text-center space-y-3">
            <FileText size={isMobile ? 36 : 48} className="mx-auto text-muted-foreground/20" />
            <h2 className={`font-semibold text-muted-foreground ${isMobile ? 'text-base' : 'text-lg'}`}>
              No Note Selected
            </h2>
            <p className={`text-muted-foreground/70 ${isMobile ? 'text-xs' : 'text-sm'}`}>
              Create or select a note to start writing
            </p>
          </div>
        </div>
      );
    }

    return (
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className={isMobile ? 'px-4 pt-3 pb-1' : 'px-8 pt-6 pb-2'}>
          <input
            value={title}
            onChange={(e) => onTitleChange(e.target.value)}
            placeholder="Note title..."
            className={cn(
              'w-full font-bold bg-transparent border-none outline-none placeholder:text-muted-foreground/40',
              isMobile ? 'text-xl' : 'text-3xl',
            )}
          />
          <div className={cn(
            'flex items-center mt-1.5 ml-1 text-muted-foreground/60',
            isMobile ? 'gap-2 text-[9px] flex-wrap' : 'gap-4 text-[11px]',
          )}>
            {createdAt && (
              <span className="flex items-center gap-1">
                <CalendarPlus size={isMobile ? 8 : 10} />
                Created: {formatTimestamp(createdAt)}
              </span>
            )}
            {updatedAt && (
              <span className="flex items-center gap-1">
                <Clock size={isMobile ? 8 : 10} />
                Modified: {formatTimestamp(updatedAt)}
              </span>
            )}
          </div>
        </div>

        {editor && (
          <EditorToolbar
            editor={editor}
            searchOpen={searchOpen}
            toggleSearch={toggleSearch}
            copied={copied}
            copyPlainText={copyPlainText}
            exportOpen={exportOpen}
            setExportOpen={setExportOpen}
            setExportError={setExportError}
            fontSize={fontSize}
            changeFontSize={changeFontSize}
            resetFontSize={resetFontSize}
            isMobile={isMobile}
          />
        )}

        {searchOpen && (
          <div className={cn(
            'flex items-center gap-2 py-2 border-b border-border bg-muted/30',
            isMobile ? 'px-3' : 'px-8',
          )}>
            <Search size={13} className="text-muted-foreground shrink-0" />
            <input
              ref={searchInputRef}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') { e.shiftKey ? prevMatch() : nextMatch(); }
                if (e.key === 'Escape') clearSearch();
              }}
              placeholder="Find in note..."
              className="flex-1 bg-transparent text-[12px] outline-none placeholder:text-muted-foreground/40"
            />
            {searchQuery && matchCount > 0 && (
              <span className="text-[10px] text-muted-foreground shrink-0">
                {matchIndex + 1}/{matchCount}{matchCount >= SEARCH_MAX_MATCHES ? '+' : ''}
              </span>
            )}
            {searchQuery && matchCount === 0 && (
              <span className="text-[10px] text-muted-foreground/50 shrink-0">No matches</span>
            )}
            <button onClick={prevMatch} disabled={matchCount === 0} className="p-0.5 rounded hover:bg-accent text-muted-foreground disabled:opacity-30">
              <ChevronUp size={14} />
            </button>
            <button onClick={nextMatch} disabled={matchCount === 0} className="p-0.5 rounded hover:bg-accent text-muted-foreground disabled:opacity-30">
              <ChevronDown size={14} />
            </button>
            <button onClick={clearSearch} className="p-0.5 rounded hover:bg-accent text-muted-foreground">
              <X size={14} />
            </button>
          </div>
        )}

        <div
          className={cn('flex-1 overflow-y-auto py-4 cursor-text', isMobile ? 'px-4' : 'px-8')}
          style={{ '--editor-font-size': `${fontSize}px` } as React.CSSProperties}
          onClick={(e) => {
            if (editor && e.target === e.currentTarget) {
              editor.commands.focus('end');
            }
          }}
        >
          <EditorContent editor={editor} />
        </div>

        {exportOpen && (
          <div
            className="fixed inset-0 z-[9999] flex items-center justify-center"
            onMouseDown={(e) => {
              if (e.target === e.currentTarget) {
                setExportOpen(false);
                setExportError(null);
              }
            }}
          >
            <div className="absolute inset-0 bg-black/40" />
            <div className="relative bg-card border border-border rounded-xl shadow-2xl w-[320px] max-w-[90vw] animate-in fade-in-0 zoom-in-95">
              <div className="flex items-center justify-between px-5 pt-5 pb-3">
                <h3 className="text-sm font-semibold text-foreground">Export Note</h3>
                <button
                  onClick={() => { setExportOpen(false); setExportError(null); }}
                  className="p-1 rounded-md text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
                >
                  <X size={16} />
                </button>
              </div>
              <div className="px-5 pb-4 space-y-2">
                <button
                  onClick={() => handleExport('pdf')}
                  disabled={!!exporting}
                  className="flex items-center gap-3 w-full px-4 py-3 rounded-lg text-sm text-foreground bg-muted/40 hover:bg-accent transition-colors disabled:opacity-50"
                >
                  <FileText size={18} className="text-red-500 shrink-0" />
                  <div className="text-left">
                    <div className="font-medium">{exporting === 'pdf' ? 'Exporting...' : 'Save as PDF'}</div>
                    <div className="text-[11px] text-muted-foreground">Portable document format</div>
                  </div>
                </button>
                <button
                  onClick={() => handleExport('docx')}
                  disabled={!!exporting}
                  className="flex items-center gap-3 w-full px-4 py-3 rounded-lg text-sm text-foreground bg-muted/40 hover:bg-accent transition-colors disabled:opacity-50"
                >
                  <FileType size={18} className="text-blue-500 shrink-0" />
                  <div className="text-left">
                    <div className="font-medium">{exporting === 'docx' ? 'Exporting...' : 'Save as DOCX'}</div>
                    <div className="text-[11px] text-muted-foreground">Microsoft Word document</div>
                  </div>
                </button>
                <button
                  onClick={() => handleExport('png')}
                  disabled={!!exporting}
                  className="flex items-center gap-3 w-full px-4 py-3 rounded-lg text-sm text-foreground bg-muted/40 hover:bg-accent transition-colors disabled:opacity-50"
                >
                  <FileImage size={18} className="text-green-500 shrink-0" />
                  <div className="text-left">
                    <div className="font-medium">{exporting === 'png' ? 'Exporting...' : 'Save as Image'}</div>
                    <div className="text-[11px] text-muted-foreground">PNG screenshot of note</div>
                  </div>
                </button>
              </div>
              {exportError && (
                <div className="mx-5 mb-4 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-[12px]">
                  {exportError}
                </div>
              )}
            </div>
          </div>
        )}

        {exporting && (
          <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-3 bg-card px-8 py-6 rounded-xl border border-border shadow-2xl">
              <Loader2 size={24} className="animate-spin text-primary" />
              <p className="text-sm font-medium text-foreground">
                Exporting as {exporting.toUpperCase()}...
              </p>
              <p className="text-[11px] text-muted-foreground">This may take a moment</p>
            </div>
          </div>
        )}
      </div>
    );
  },
);

export default NoteEditor;

// Heading: if the user selects partial text within a block, split the block
// at selection boundaries first so only the selected portion becomes a heading.
// Without this, the entire paragraph converts (standard but unintuitive).
function applyHeading(e: ReturnType<typeof useEditor> & {}, level: 1 | 2 | 3) {
  const { from, to, empty } = e.state.selection;

  if (empty) {
    e.chain().focus().toggleHeading({ level }).run();
    return;
  }

  const $from = e.state.doc.resolve(from);
  const $to = e.state.doc.resolve(to);

  // Multi-block selection or entire block selected — standard toggle
  if (!$from.sameParent($to) || (from <= $from.start() && to >= $from.end())) {
    e.chain().focus().toggleHeading({ level }).run();
    return;
  }

  // Partial selection within one block — split, then heading
  const blockEnd = $from.end();

  if (to < blockEnd) {
    e.commands.setTextSelection(to);
    e.commands.splitBlock();
  }
  // from is before the split point, so position is still valid
  if (from > $from.start()) {
    e.commands.setTextSelection(from);
    e.commands.splitBlock();
  }
  // Cursor is now in the middle block (the selected text)
  e.commands.toggleHeading({ level });
}

// Toolbar — prevents 20+ button re-renders per keystroke

interface EditorToolbarProps {
  editor: ReturnType<typeof useEditor> & {};
  searchOpen: boolean;
  toggleSearch: () => void;
  copied: boolean;
  copyPlainText: () => void;
  exportOpen: boolean;
  setExportOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setExportError: React.Dispatch<React.SetStateAction<string | null>>;
  fontSize: number;
  changeFontSize: (delta: number) => void;
  resetFontSize: () => void;
  isMobile?: boolean;
}

const EditorToolbar = function EditorToolbar({
  editor, searchOpen, toggleSearch, copied, copyPlainText,
  exportOpen, setExportOpen, setExportError,
  fontSize, changeFontSize, resetFontSize, isMobile,
}: EditorToolbarProps) {
  const editorRef = useRef(editor);
  editorRef.current = editor;

  // Stable dispatcher ref — lets memo'd ToolBtns skip re-renders
  const exec = useCallback((cmd: string) => {
    const e = editorRef.current;
    if (!e) return;
    switch (cmd) {
      case 'bold': e.chain().focus().toggleBold().run(); break;
      case 'italic': e.chain().focus().toggleItalic().run(); break;
      case 'underline': e.chain().focus().toggleMark('underline').run(); break;
      case 'strike': e.chain().focus().toggleStrike().run(); break;
      case 'code': e.chain().focus().toggleCode().run(); break;
      case 'highlight': e.chain().focus().toggleHighlight().run(); break;
      case 'align-left': e.chain().focus().setTextAlign('left').run(); break;
      case 'align-center': e.chain().focus().setTextAlign('center').run(); break;
      case 'align-right': e.chain().focus().setTextAlign('right').run(); break;
      case 'h1': applyHeading(e, 1); break;
      case 'h2': applyHeading(e, 2); break;
      case 'h3': applyHeading(e, 3); break;
      case 'bullet': e.chain().focus().toggleBulletList().run(); break;
      case 'ordered': e.chain().focus().toggleOrderedList().run(); break;
      case 'task': e.chain().focus().toggleTaskList().run(); break;
      case 'quote': e.chain().focus().toggleBlockquote().run(); break;
      case 'hr': e.chain().focus().setHorizontalRule().run(); break;
      case 'codeblock': e.chain().focus().toggleCodeBlock().run(); break;
    }
  }, []);

  return (
    <div className={cn(
      'flex items-center gap-0.5 py-2 border-b border-border overflow-x-auto',
      isMobile ? 'px-3' : 'px-8 flex-wrap',
    )}>
      <ToolBtn cmd="bold" exec={exec} active={editor.isActive('bold')} title={`Bold (${MOD}+B)`}><Bold size={14} /></ToolBtn>
      <ToolBtn cmd="italic" exec={exec} active={editor.isActive('italic')} title={`Italic (${MOD}+I)`}><Italic size={14} /></ToolBtn>
      <ToolBtn cmd="underline" exec={exec} active={editor.isActive('underline')} title={`Underline (${MOD}+U)`}><UnderlineIcon size={14} /></ToolBtn>
      <ToolBtn cmd="strike" exec={exec} active={editor.isActive('strike')} title={`Strikethrough (${MOD}+Shift+X)`}><Strikethrough size={14} /></ToolBtn>
      <ToolBtn cmd="code" exec={exec} active={editor.isActive('code')} title={`Inline code (${MOD}+E)`}><Code size={14} /></ToolBtn>
      <ToolBtn cmd="highlight" exec={exec} active={editor.isActive('highlight')} title="Highlight"><Highlighter size={14} /></ToolBtn>
      <div className="w-px h-5 bg-border mx-1 shrink-0" />
      <ToolBtn cmd="align-left" exec={exec} active={editor.isActive({ textAlign: 'left' })} title="Align left"><AlignLeft size={14} /></ToolBtn>
      <ToolBtn cmd="align-center" exec={exec} active={editor.isActive({ textAlign: 'center' })} title="Align center"><AlignCenter size={14} /></ToolBtn>
      <ToolBtn cmd="align-right" exec={exec} active={editor.isActive({ textAlign: 'right' })} title="Align right"><AlignRight size={14} /></ToolBtn>
      <div className="w-px h-5 bg-border mx-1 shrink-0" />
      <ToolBtn cmd="h1" exec={exec} active={editor.isActive('heading', { level: 1 })} title={`Heading 1 (${MOD}+${ALT}+1)`}><Heading1 size={14} /></ToolBtn>
      <ToolBtn cmd="h2" exec={exec} active={editor.isActive('heading', { level: 2 })} title={`Heading 2 (${MOD}+${ALT}+2)`}><Heading2 size={14} /></ToolBtn>
      <ToolBtn cmd="h3" exec={exec} active={editor.isActive('heading', { level: 3 })} title={`Heading 3 (${MOD}+${ALT}+3)`}><Heading3 size={14} /></ToolBtn>
      <div className="w-px h-5 bg-border mx-1 shrink-0" />
      <ToolBtn cmd="bullet" exec={exec} active={editor.isActive('bulletList')} title={`Bullet list (${MOD}+Shift+8)`}><List size={14} /></ToolBtn>
      <ToolBtn cmd="ordered" exec={exec} active={editor.isActive('orderedList')} title={`Numbered list (${MOD}+Shift+7)`}><ListOrdered size={14} /></ToolBtn>
      <ToolBtn cmd="task" exec={exec} active={editor.isActive('taskList')} title="Task list"><CheckSquare size={14} /></ToolBtn>
      <ToolBtn cmd="quote" exec={exec} active={editor.isActive('blockquote')} title={`Quote (${MOD}+Shift+B)`}><Quote size={14} /></ToolBtn>
      <ToolBtn cmd="hr" exec={exec} active={false} title="Horizontal rule"><Minus size={14} /></ToolBtn>
      <ToolBtn cmd="codeblock" exec={exec} active={editor.isActive('codeBlock')} title={`Code block (${MOD}+${ALT}+C)`}><Code size={14} /></ToolBtn>
      <div className="w-px h-5 bg-border mx-1 shrink-0" />
      <button
        onMouseDown={(e) => { e.preventDefault(); toggleSearch(); }}
        className={cn(
          'flex items-center gap-1 px-2 py-1.5 rounded-md text-sm shrink-0',
          searchOpen ? 'text-primary bg-primary/10' : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
        )}
        title="Search in note"
      >
        <Search size={14} />
      </button>
      <button
        onMouseDown={(e) => { e.preventDefault(); copyPlainText(); }}
        className={cn(
          'flex items-center gap-1 px-2 py-1.5 rounded-md text-sm shrink-0',
          copied ? 'text-green-500 bg-green-500/10' : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
        )}
        title={copied ? 'Copied!' : 'Copy text'}
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
      <button
        onMouseDown={(e) => { e.preventDefault(); preloadExportLibs(); setExportOpen(prev => !prev); setExportError(null); }}
        className={cn(
          'flex items-center gap-1 px-2 py-1.5 rounded-md text-sm shrink-0',
          exportOpen ? 'text-primary bg-primary/10' : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
        )}
        title="Export / Save As"
      >
        <Download size={14} />
      </button>
      <div className="w-px h-5 bg-border mx-1 shrink-0" />
      <button
        onMouseDown={(e) => { e.preventDefault(); changeFontSize(-FONT_SIZE_STEP); }}
        disabled={fontSize <= FONT_SIZE_MIN}
        className="flex items-center px-1.5 py-1.5 rounded-md text-sm shrink-0 text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-30"
        title="Decrease font size (Cmd/Ctrl + −)"
      >
        <AArrowDown size={14} />
      </button>
      <button
        onMouseDown={(e) => { e.preventDefault(); resetFontSize(); }}
        className="flex items-center px-1 py-1.5 rounded-md text-[10px] font-mono shrink-0 text-muted-foreground hover:bg-accent hover:text-accent-foreground"
        title="Reset font size (Cmd/Ctrl + 0)"
      >
        {fontSize}
      </button>
      <button
        onMouseDown={(e) => { e.preventDefault(); changeFontSize(FONT_SIZE_STEP); }}
        disabled={fontSize >= FONT_SIZE_MAX}
        className="flex items-center px-1.5 py-1.5 rounded-md text-sm shrink-0 text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-30"
        title="Increase font size (Cmd/Ctrl + +)"
      >
        <AArrowUp size={14} />
      </button>
    </div>
  );
};

// ToolBtn — memo skips re-render unless `active` changes

const ToolBtn = memo(function ToolBtn(
  { cmd, exec, active, children, title }: {
    cmd: string; exec: (cmd: string) => void;
    active: boolean; children: React.ReactNode; title?: string;
  },
) {
  return (
    <button
      onMouseDown={(e) => { e.preventDefault(); exec(cmd); }}
      title={title}
      className={cn(
        'flex items-center gap-1 px-2 py-1.5 rounded-md text-sm shrink-0',
        active ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
      )}
    >
      {children}
    </button>
  );
}, (prev, next) => prev.active === next.active && prev.cmd === next.cmd);

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}
