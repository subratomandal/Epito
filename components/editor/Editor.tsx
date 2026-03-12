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
  Download, FileImage, FileType,
  AlignLeft, AlignCenter, AlignRight,
} from 'lucide-react';
import { exportAsPDF, exportAsDOCX, exportAsImage } from '@/lib/export';
import type { ExportFormat } from '@/lib/export';

const lowlight = createLowlight(common);

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.platform);
const MOD = IS_MAC ? '⌘' : 'Ctrl';
const ALT = IS_MAC ? '⌥' : 'Alt';

const PASTE_HTML_MAX = 200_000;
const PASTE_TEXT_MAX = 5_000_000;
const SEARCH_DEBOUNCE = 250;
const SEARCH_MAX_MATCHES = 500;

const FONT_SIZE_KEY = 'epito-editor-font-size';
const FONT_SIZE_DEFAULT = 16;
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
  return text.length > PASTE_TEXT_MAX ? text.slice(0, PASTE_TEXT_MAX) : text;
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
      TiptapLink.configure({ openOnClick: false }),
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

      if (content !== loadedContentRef.current) {
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
      setExporting(format);
      setExportError(null);
      try {
        const html = editor.getHTML();
        const noteTitle = title || 'Untitled';
        if (format === 'pdf') await exportAsPDF(html, noteTitle);
        else if (format === 'docx') await exportAsDOCX(html, noteTitle);
        else if (format === 'png') await exportAsImage(html, noteTitle);
        setExportOpen(false);
      } catch (err) {
        console.error(`Export as ${format} failed:`, err);
        setExportError(
          err instanceof Error ? err.message : `Failed to export as ${format.toUpperCase()}. Please try again.`
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
          <div className={cn(
            'flex items-center gap-0.5 py-2 border-b border-border overflow-x-auto',
            isMobile ? 'px-3' : 'px-8 flex-wrap',
          )}>
            <ToolBtn onClick={() => editor.chain().focus().toggleBold().run()} active={editor.isActive('bold')} title={`Bold (${MOD}+B)`}><Bold size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleItalic().run()} active={editor.isActive('italic')} title={`Italic (${MOD}+I)`}><Italic size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleMark('underline').run()} active={editor.isActive('underline')} title={`Underline (${MOD}+U)`}><UnderlineIcon size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleStrike().run()} active={editor.isActive('strike')} title={`Strikethrough (${MOD}+Shift+X)`}><Strikethrough size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleCode().run()} active={editor.isActive('code')} title={`Inline code (${MOD}+E)`}><Code size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleHighlight().run()} active={editor.isActive('highlight')} title="Highlight"><Highlighter size={14} /></ToolBtn>
            <div className="w-px h-5 bg-border mx-1 shrink-0" />
            <ToolBtn onClick={() => editor.chain().focus().setTextAlign('left').run()} active={editor.isActive({ textAlign: 'left' })} title="Align left"><AlignLeft size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().setTextAlign('center').run()} active={editor.isActive({ textAlign: 'center' })} title="Align center"><AlignCenter size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().setTextAlign('right').run()} active={editor.isActive({ textAlign: 'right' })} title="Align right"><AlignRight size={14} /></ToolBtn>
            <div className="w-px h-5 bg-border mx-1 shrink-0" />
            <ToolBtn onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()} active={editor.isActive('heading', { level: 1 })} title={`Heading 1 (${MOD}+${ALT}+1)`}><Heading1 size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()} active={editor.isActive('heading', { level: 2 })} title={`Heading 2 (${MOD}+${ALT}+2)`}><Heading2 size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()} active={editor.isActive('heading', { level: 3 })} title={`Heading 3 (${MOD}+${ALT}+3)`}><Heading3 size={14} /></ToolBtn>
            <div className="w-px h-5 bg-border mx-1 shrink-0" />
            <ToolBtn onClick={() => editor.chain().focus().toggleBulletList().run()} active={editor.isActive('bulletList')} title={`Bullet list (${MOD}+Shift+8)`}><List size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleOrderedList().run()} active={editor.isActive('orderedList')} title={`Numbered list (${MOD}+Shift+7)`}><ListOrdered size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleTaskList().run()} active={editor.isActive('taskList')} title="Task list"><CheckSquare size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleBlockquote().run()} active={editor.isActive('blockquote')} title={`Quote (${MOD}+Shift+B)`}><Quote size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().setHorizontalRule().run()} active={false} title="Horizontal rule"><Minus size={14} /></ToolBtn>
            <ToolBtn onClick={() => editor.chain().focus().toggleCodeBlock().run()} active={editor.isActive('codeBlock')} title={`Code block (${MOD}+${ALT}+C)`}><Code size={14} /></ToolBtn>
            <div className="w-px h-5 bg-border mx-1 shrink-0" />
            <button
              onMouseDown={(e) => { e.preventDefault(); toggleSearch(); }}
              className={cn(
                'flex items-center gap-1 px-2 py-1.5 rounded-md text-sm transition-colors shrink-0',
                searchOpen ? 'text-primary bg-primary/10' : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
              )}
              title="Search in note"
            >
              <Search size={14} />
            </button>
            <button
              onMouseDown={(e) => { e.preventDefault(); copyPlainText(); }}
              className={cn(
                'flex items-center gap-1 px-2 py-1.5 rounded-md text-sm transition-colors shrink-0',
                copied ? 'text-green-500 bg-green-500/10' : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
              )}
              title={copied ? 'Copied!' : 'Copy text'}
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
            <button
              onMouseDown={(e) => { e.preventDefault(); setExportOpen(prev => !prev); setExportError(null); }}
              className={cn(
                'flex items-center gap-1 px-2 py-1.5 rounded-md text-sm transition-colors shrink-0',
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
              className="flex items-center px-1.5 py-1.5 rounded-md text-sm transition-colors shrink-0 text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-30"
              title="Decrease font size (Cmd/Ctrl + −)"
            >
              <AArrowDown size={14} />
            </button>
            <button
              onMouseDown={(e) => { e.preventDefault(); resetFontSize(); }}
              className="flex items-center px-1 py-1.5 rounded-md text-[10px] font-mono transition-colors shrink-0 text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              title="Reset font size (Cmd/Ctrl + 0)"
            >
              {fontSize}
            </button>
            <button
              onMouseDown={(e) => { e.preventDefault(); changeFontSize(FONT_SIZE_STEP); }}
              disabled={fontSize >= FONT_SIZE_MAX}
              className="flex items-center px-1.5 py-1.5 rounded-md text-sm transition-colors shrink-0 text-muted-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-30"
              title="Increase font size (Cmd/Ctrl + +)"
            >
              <AArrowUp size={14} />
            </button>
          </div>
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
      </div>
    );
  },
);

export default NoteEditor;

const ToolBtn = memo(function ToolBtn(
  { onClick, active, children, title }: { onClick: () => void; active: boolean; children: React.ReactNode; title?: string },
) {
  return (
    <button
      onMouseDown={(e) => { e.preventDefault(); onClick(); }}
      title={title}
      className={cn(
        'flex items-center gap-1 px-2 py-1.5 rounded-md text-sm transition-colors shrink-0',
        active ? 'bg-accent text-accent-foreground' : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
      )}
    >
      {children}
    </button>
  );
});

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}
