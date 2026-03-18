'use client';

import { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import { Button, Input, ScrollArea, Separator, Tooltip } from '@/components/ui/primitives';
import {
  Plus, Search, FileText, Trash2,
  FileUp, ImagePlus, File, Image as ImageIcon, Loader2,
  PanelLeftClose, PanelLeft, StickyNote,
  GripVertical, ChevronDown, ChevronRight, RotateCcw, Settings,
  Sun, Moon, Check,
} from 'lucide-react';
import type { Note, Document, UploadedImage } from '@/common/types';

interface SidebarProps {
  notes: Note[];
  documents: Document[];
  images: UploadedImage[];
  selectedId: string | null;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onSelect: (note: Note) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onSearchOpen: () => void;
  onUploadDocument: (file: File) => void;
  onUploadImage: (file: File) => void;
  onViewDocument: (doc: Document) => void;
  onViewImage: (img: UploadedImage) => void;
  onDeleteDocument: (id: string) => void;
  onDeleteImage: (id: string) => void;
  deletedNotes: Note[];
  onRestoreNote: (id: string) => void;
  onPermanentDelete: (id: string) => void;
  theme: 'light' | 'dark';
  onThemeChange: (theme: 'light' | 'dark') => void;
  uploadingDoc: boolean;
  uploadingImg: boolean;
}

export default function Sidebar({
  notes, documents, images, selectedId, collapsed, onToggleCollapse,
  onSelect, onCreate, onDelete, onSearchOpen,
  onUploadDocument, onUploadImage, onViewDocument, onViewImage,
  onDeleteDocument, onDeleteImage,
  deletedNotes, onRestoreNote, onPermanentDelete,
  theme, onThemeChange,
  uploadingDoc, uploadingImg,
}: SidebarProps) {
  const [filter, setFilter] = useState('');
  const [viewMode, setViewMode] = useState<'notes' | 'docs' | 'images'>('notes');
  const [docDragOver, setDocDragOver] = useState(false);
  const [imgDragOver, setImgDragOver] = useState(false);
  const [deletedExpanded, setDeletedExpanded] = useState(false);
  const [settingsExpanded, setSettingsExpanded] = useState(false);
  const docInputRef = useRef<HTMLInputElement>(null);
  const imgInputRef = useRef<HTMLInputElement>(null);
  const noteListRef = useRef<HTMLDivElement>(null);
  const [dragState, setDragState] = useState<{ srcIdx: number; overIdx: number } | null>(null);

  const [noteOrder, setNoteOrder] = useState<string[]>([]);
  const orderLoadedRef = useRef(false);

  // Server-persisted order survives port changes and app restarts
  useEffect(() => {
    fetch('/api/settings?key=note-order')
      .then(r => r.ok ? r.json() : { value: null })
      .then(data => {
        if (Array.isArray(data.value) && data.value.length > 0) {
          setNoteOrder(data.value);
        }
        orderLoadedRef.current = true;
      })
      .catch(() => { orderLoadedRef.current = true; });
  }, []);

  // Persist order changes (skip initial load)
  useEffect(() => {
    if (!orderLoadedRef.current || noteOrder.length === 0) return;
    fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: 'note-order', value: noteOrder }),
    }).catch(() => {});
  }, [noteOrder]);

  const orderedNotes = useMemo(() => {
    if (noteOrder.length === 0) return notes;
    const orderMap = new Map(noteOrder.map((id, i) => [id, i]));
    return [...notes].sort((a, b) => {
      const oa = orderMap.get(a.id);
      const ob = orderMap.get(b.id);
      if (oa !== undefined && ob !== undefined) return oa - ob;
      if (oa !== undefined) return -1;
      if (ob !== undefined) return 1;
      return 0;
    });
  }, [notes, noteOrder]);

  const startNoteDrag = useCallback((srcIdx: number, e: React.PointerEvent) => {
    if (filter) return;
    e.preventDefault();
    const startY = e.clientY;
    const listEl = noteListRef.current;
    if (!listEl) return;
    const items = Array.from(listEl.children) as HTMLElement[];
    const itemHeight = items[0]?.offsetHeight || 32;
    const count = items.length;

    setDragState({ srcIdx, overIdx: srcIdx });

    const onMove = (ev: PointerEvent) => {
      const dy = ev.clientY - startY;
      const offset = Math.round(dy / itemHeight);
      const overIdx = Math.max(0, Math.min(count - 1, srcIdx + offset));
      setDragState({ srcIdx, overIdx });
    };

    const onUp = () => {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);

      setDragState(prev => {
        if (prev && prev.srcIdx !== prev.overIdx) {
          const ids = orderedNotes.map(n => n.id);
          const [moved] = ids.splice(prev.srcIdx, 1);
          ids.splice(prev.overIdx, 0, moved);
          setTimeout(() => setNoteOrder(ids), 0);
        }
        return null;
      });
    };

    document.addEventListener('pointermove', onMove);
    document.addEventListener('pointerup', onUp);
  }, [orderedNotes, filter]);

  const filtered = useMemo(() => {
    if (!filter) return orderedNotes;
    const q = filter.toLowerCase();
    return orderedNotes.filter(n => n.title.toLowerCase().includes(q));
  }, [orderedNotes, filter]);

  const filteredDocs = useMemo(() => {
    if (!filter) return documents;
    const q = filter.toLowerCase();
    return documents.filter(d => d.filename.toLowerCase().includes(q));
  }, [documents, filter]);

  const filteredImages = useMemo(() => {
    if (!filter) return images;
    const q = filter.toLowerCase();
    return images.filter(i => i.filename.toLowerCase().includes(q));
  }, [images, filter]);

  const viewModes = [
    { key: 'notes' as const, icon: StickyNote, label: 'Notes' },
    { key: 'docs' as const, icon: FileText, label: 'Docs' },
    { key: 'images' as const, icon: ImageIcon, label: 'Images' },
  ];

  return (
    <aside
      className="flex flex-col bg-card border-r border-border shrink-0 h-full overflow-hidden transition-[width] duration-200 ease-out"
      style={{ width: collapsed ? 48 : 256 }}
    >
      {collapsed ? (
        <>
          <div className="p-1.5 flex flex-col items-center">
            <Tooltip content="Expand sidebar" side="right">
              <button onClick={onToggleCollapse} className="w-8 h-8 rounded-md flex items-center justify-center hover:bg-accent transition-colors text-muted-foreground hover:text-foreground">
                <PanelLeft size={16} />
              </button>
            </Tooltip>
          </div>

          <Separator />

          <div className="p-1.5 space-y-1 flex flex-col items-center">
            <Tooltip content="Search (Cmd+K)" side="right">
              <Button variant="ghost" size="icon" onClick={onSearchOpen} className="w-8 h-8">
                <Search size={14} />
              </Button>
            </Tooltip>
            <Tooltip content="New Note (Cmd+N)" side="right">
              <Button variant="ghost" size="icon" onClick={onCreate} className="w-8 h-8">
                <Plus size={14} />
              </Button>
            </Tooltip>
          </div>

          <Separator />

          <div className="flex-1 flex flex-col items-center py-2 space-y-1">
            {viewModes.map(({ key, icon: Icon, label }) => (
              <Tooltip key={key} content={label} side="right">
                <button
                  onClick={() => setViewMode(key)}
                  className={`w-8 h-8 rounded-md flex items-center justify-center transition-colors ${
                    viewMode === key ? 'bg-accent text-foreground' : 'text-muted-foreground hover:bg-accent hover:text-foreground'
                  }`}
                >
                  <Icon size={14} />
                </button>
              </Tooltip>
            ))}
          </div>

          <Separator />

          <div className="p-1.5 flex flex-col items-center space-y-1">
            <Tooltip content="Upload Document" side="right">
              <button
                onClick={() => docInputRef.current?.click()}
                className="w-8 h-8 rounded-md flex items-center justify-center text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
              >
                {uploadingDoc ? <Loader2 size={14} className="animate-spin text-primary" /> : <FileUp size={14} />}
              </button>
            </Tooltip>
            <Tooltip content="Upload Image" side="right">
              <button
                onClick={() => imgInputRef.current?.click()}
                className="w-8 h-8 rounded-md flex items-center justify-center text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
              >
                {uploadingImg ? <Loader2 size={14} className="animate-spin text-primary" /> : <ImagePlus size={14} />}
              </button>
            </Tooltip>
          </div>

          <Separator />

          <div className="p-1.5 flex flex-col items-center">
            <Tooltip content="Settings" side="right">
              <button
                onClick={() => onThemeChange(theme === 'dark' ? 'light' : 'dark')}
                className="w-8 h-8 rounded-md flex items-center justify-center text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
              >
                {theme === 'dark' ? <Moon size={14} /> : <Sun size={14} />}
              </button>
            </Tooltip>
          </div>

          <input ref={docInputRef} type="file" accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onUploadDocument(f); e.target.value = ''; }} />
          <input ref={imgInputRef} type="file" accept=".png,.jpg,.jpeg,image/png,image/jpeg" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onUploadImage(f); e.target.value = ''; }} />
        </>
      ) : (
        <>
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <h1 className="text-base font-bold tracking-tight text-foreground">Epito</h1>
          <div className="flex gap-0.5">
            <Button variant="ghost" size="icon" onClick={onSearchOpen} title="Search (Cmd+K)" className="h-7 w-7">
              <Search size={14} />
            </Button>
            <Button variant="ghost" size="icon" onClick={onCreate} title="New Note (Cmd+N)" className="h-7 w-7">
              <Plus size={14} />
            </Button>
            <Button variant="ghost" size="icon" onClick={onToggleCollapse} title="Collapse sidebar" className="h-7 w-7">
              <PanelLeftClose size={14} />
            </Button>
          </div>
        </div>

        <Input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Search..."
          className="h-7 text-xs"
        />

        <div className="flex gap-1 p-0.5 bg-muted rounded-lg">
          {viewModes.map(({ key, icon: Icon, label }) => (
            <button
              key={key}
              onClick={() => setViewMode(key)}
              className={`flex-1 flex items-center justify-center gap-1.5 text-[11px] py-1.5 rounded-md font-medium transition-all duration-150 ${
                viewMode === key
                  ? 'bg-card text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Icon size={11} />
              {label}
            </button>
          ))}
        </div>
      </div>

      <Separator />

      <ScrollArea className="flex-1">
        <div className="px-3 py-3">
          <div className="bg-muted rounded-lg p-1 space-y-0.5">
          {viewMode === 'notes' && (
            <>
              {filtered.length === 0 ? (
                <div className="text-center py-8 space-y-2">
                  <StickyNote size={24} className="mx-auto text-muted-foreground/20" />
                  <p className="text-xs text-muted-foreground/50">
                    {filter ? 'No matching notes' : 'No notes yet'}
                  </p>
                </div>
              ) : (
                <div ref={noteListRef}>
                  {filtered.map((note, idx) => {
                    let offset = 0;
                    let isDragging = false;
                    if (dragState && !filter) {
                      const { srcIdx, overIdx } = dragState;
                      if (idx === srcIdx) {
                        offset = overIdx - srcIdx;
                        isDragging = true;
                      } else if (srcIdx < overIdx && idx > srcIdx && idx <= overIdx) {
                        offset = -1;
                      } else if (srcIdx > overIdx && idx < srcIdx && idx >= overIdx) {
                        offset = 1;
                      }
                    }
                    return (
                      <div
                        key={note.id}
                        style={dragState ? {
                          transform: `translateY(${offset * 100}%)`,
                          transition: isDragging ? 'transform 50ms ease' : 'transform 150ms ease',
                          zIndex: isDragging ? 10 : 1,
                          position: 'relative',
                          opacity: isDragging ? 0.8 : 1,
                        } : undefined}
                      >
                        <NoteItem
                          note={note}
                          selected={selectedId === note.id}
                          onSelect={() => onSelect(note)}
                          onDelete={() => onDelete(note.id)}
                          onGripDown={!filter ? (e) => startNoteDrag(idx, e) : undefined}
                        />
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}

          {viewMode === 'docs' && (
            <div className="space-y-2">
              <div
                onDragOver={(e) => { e.preventDefault(); setDocDragOver(true); }}
                onDragLeave={() => setDocDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDocDragOver(false);
                  const file = e.dataTransfer?.files?.[0];
                  if (file && (file.type === 'application/pdf' || file.name.endsWith('.pdf') || file.name.endsWith('.docx'))) {
                    onUploadDocument(file);
                  }
                }}
              >
                <button
                  onClick={() => docInputRef.current?.click()}
                  disabled={uploadingDoc}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg border border-dashed transition-all duration-150 ${
                    docDragOver
                      ? 'border-primary bg-primary/10 text-primary'
                      : uploadingDoc
                        ? 'border-primary/50 bg-primary/5 text-primary'
                        : 'border-border text-muted-foreground hover:border-primary/50 hover:bg-accent hover:text-foreground'
                  }`}
                >
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-colors ${
                    docDragOver || uploadingDoc ? 'bg-primary/20' : 'bg-muted'
                  }`}>
                    {uploadingDoc ? (
                      <Loader2 size={16} className="animate-spin text-primary" />
                    ) : (
                      <FileUp size={16} className={docDragOver ? 'text-primary' : 'text-muted-foreground'} />
                    )}
                  </div>
                  <div className="text-left">
                    <p className="text-xs font-medium">{uploadingDoc ? 'Uploading...' : 'Upload Document'}</p>
                    <p className="text-[10px] text-muted-foreground/60">PDF or DOCX</p>
                  </div>
                </button>
              </div>
              <input
                ref={docInputRef}
                type="file"
                accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onUploadDocument(f);
                  e.target.value = '';
                }}
              />

              {filteredDocs.length === 0 ? (
                <div className="text-center py-6 space-y-2">
                  <FileText size={24} className="mx-auto text-muted-foreground/20" />
                  <p className="text-xs text-muted-foreground/50">No documents yet</p>
                </div>
              ) : filteredDocs.map(doc => (
                <DocItem
                  key={doc.id}
                  doc={doc}
                  selected={selectedId === doc.id}
                  onView={() => onViewDocument(doc)}
                  onDelete={() => onDeleteDocument(doc.id)}
                />
              ))}
            </div>
          )}

          {viewMode === 'images' && (
            <div className="space-y-2">
              <div
                onDragOver={(e) => { e.preventDefault(); setImgDragOver(true); }}
                onDragLeave={() => setImgDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setImgDragOver(false);
                  const file = e.dataTransfer?.files?.[0];
                  if (file && file.type.startsWith('image/')) {
                    onUploadImage(file);
                  }
                }}
              >
                <button
                  onClick={() => imgInputRef.current?.click()}
                  disabled={uploadingImg}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg border border-dashed transition-all duration-150 ${
                    imgDragOver
                      ? 'border-primary bg-primary/10 text-primary'
                      : uploadingImg
                        ? 'border-primary/50 bg-primary/5 text-primary'
                        : 'border-border text-muted-foreground hover:border-primary/50 hover:bg-accent hover:text-foreground'
                  }`}
                >
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-colors ${
                    imgDragOver || uploadingImg ? 'bg-primary/20' : 'bg-muted'
                  }`}>
                    {uploadingImg ? (
                      <Loader2 size={16} className="animate-spin text-primary" />
                    ) : (
                      <ImagePlus size={16} className={imgDragOver ? 'text-primary' : 'text-muted-foreground'} />
                    )}
                  </div>
                  <div className="text-left">
                    <p className="text-xs font-medium">{uploadingImg ? 'Uploading...' : 'Upload Image'}</p>
                    <p className="text-[10px] text-muted-foreground/60">PNG, JPG</p>
                  </div>
                </button>
              </div>
              <input
                ref={imgInputRef}
                type="file"
                accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onUploadImage(f);
                  e.target.value = '';
                }}
              />

              {filteredImages.length === 0 ? (
                <div className="text-center py-6 space-y-2">
                  <ImageIcon size={24} className="mx-auto text-muted-foreground/20" />
                  <p className="text-xs text-muted-foreground/50">No images yet</p>
                </div>
              ) : filteredImages.map(img => (
                <ImgItem
                  key={img.id}
                  img={img}
                  selected={selectedId === img.id}
                  onView={() => onViewImage(img)}
                  onDelete={() => onDeleteImage(img.id)}
                />
              ))}
            </div>
          )}
          </div>
        </div>
      </ScrollArea>

      {deletedNotes.length > 0 && (
        <div className="border-t border-border">
          <button
            onClick={() => setDeletedExpanded(!deletedExpanded)}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
            aria-expanded={deletedExpanded}
            aria-label="Recently Deleted"
          >
            {deletedExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <Trash2 size={12} />
            <span className="font-medium">Recently Deleted</span>
            <span className="ml-auto text-[10px] bg-muted px-1.5 py-0.5 rounded-full">{deletedNotes.length}</span>
          </button>
          {deletedExpanded && (
            <ScrollArea className="max-h-40">
              <div className="px-1.5 pb-1.5 space-y-0.5">
                {deletedNotes.map(note => (
                  <div
                    key={note.id}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-muted-foreground group hover:bg-accent"
                  >
                    <FileText size={11} className="shrink-0 opacity-50" />
                    <span className="truncate flex-1">{note.title || 'Untitled'}</span>
                    <button
                      onClick={() => onRestoreNote(note.id)}
                      className="shrink-0 md:opacity-0 md:group-hover:opacity-100 text-primary hover:text-primary/80 transition-opacity"
                      title="Restore"
                      aria-label="Restore note"
                    >
                      <RotateCcw size={11} />
                    </button>
                    <button
                      onClick={() => onPermanentDelete(note.id)}
                      className="shrink-0 md:opacity-0 md:group-hover:opacity-100 text-destructive hover:text-destructive/80 transition-opacity"
                      title="Delete permanently"
                      aria-label="Delete permanently"
                    >
                      <Trash2 size={11} />
                    </button>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </div>
      )}

      <div className="border-t border-border">
        <button
          onClick={() => setSettingsExpanded(!settingsExpanded)}
          className="w-full flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
          aria-expanded={settingsExpanded}
          aria-label="Settings"
        >
          {settingsExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <Settings size={12} />
          <span className="font-medium">Settings</span>
        </button>
        {settingsExpanded && (
          <div className="px-1.5 pb-1.5 space-y-0.5">
            <button
              onClick={() => onThemeChange('light')}
              className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors ${
                theme === 'light'
                  ? 'bg-accent text-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              }`}
            >
              <Sun size={11} className="shrink-0" />
              <span className="truncate flex-1 text-left">Light</span>
              {theme === 'light' && <Check size={11} className="shrink-0 text-primary" />}
            </button>
            <button
              onClick={() => onThemeChange('dark')}
              className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors ${
                theme === 'dark'
                  ? 'bg-accent text-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              }`}
            >
              <Moon size={11} className="shrink-0" />
              <span className="truncate flex-1 text-left">Dark</span>
              {theme === 'dark' && <Check size={11} className="shrink-0 text-primary" />}
            </button>
          </div>
        )}
      </div>
      </>
      )}
    </aside>
  );
}

function NoteItem({ note, selected, onSelect, onDelete, onGripDown }: {
  note: Note; selected: boolean; onSelect: () => void; onDelete: () => void;
  onGripDown?: (e: React.PointerEvent) => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={`w-full text-left group flex items-center gap-2.5 px-2.5 py-1.5 rounded-md transition-all duration-150 cursor-pointer ${
        selected
          ? 'bg-card text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      }`}
    >
      <FileText size={14} className={`shrink-0 ${selected ? 'text-primary' : ''}`} />
      <span className="text-[13px] font-medium truncate flex-1">{note.title || 'Untitled'}</span>
      {onGripDown && (
        <div
          onPointerDown={(e) => { e.stopPropagation(); onGripDown(e); }}
          className="shrink-0 p-0.5 cursor-grab active:cursor-grabbing rounded hover:bg-accent md:opacity-0 md:group-hover:opacity-50 transition-opacity touch-none"
        >
          <GripVertical size={11} />
        </div>
      )}
      <button
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="shrink-0 md:opacity-0 md:group-hover:opacity-100 transition-opacity p-0.5"
        title="Delete"
      >
        <Trash2 size={13} className="text-muted-foreground hover:text-red-500" />
      </button>
    </div>
  );
}

function DocItem({ doc, selected, onView, onDelete }: {
  doc: Document; selected: boolean; onView: () => void; onDelete: () => void;
}) {
  return (
    <button
      onClick={onView}
      className={`w-full text-left group flex items-center gap-2.5 px-2.5 py-1.5 rounded-md transition-all duration-150 ${
        selected
          ? 'bg-card text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      }`}
    >
      <File size={14} className={`shrink-0 ${doc.file_type === 'pdf' ? 'text-red-400' : 'text-blue-400'}`} />
      <span className="text-[13px] font-medium truncate flex-1">{doc.filename}</span>
      {doc.status === 'processing' && <Loader2 size={11} className="shrink-0 animate-spin text-yellow-400" />}
      <button
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="shrink-0 md:opacity-0 md:group-hover:opacity-100 transition-opacity"
        title="Delete"
        aria-label="Delete document"
      >
        <Trash2 size={13} className="text-muted-foreground hover:text-red-500" />
      </button>
    </button>
  );
}

function ImgItem({ img, selected, onView, onDelete }: {
  img: UploadedImage; selected: boolean; onView: () => void; onDelete: () => void;
}) {
  return (
    <button
      onClick={onView}
      className={`w-full text-left group flex items-center gap-2.5 px-2.5 py-1.5 rounded-md transition-all duration-150 ${
        selected
          ? 'bg-card text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      }`}
    >
      <ImageIcon size={14} className="shrink-0 text-green-400" />
      <span className="text-[13px] font-medium truncate flex-1">{img.filename}</span>
      {img.status === 'processing' && <Loader2 size={11} className="shrink-0 animate-spin text-yellow-400" />}
      <button
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="shrink-0 md:opacity-0 md:group-hover:opacity-100 transition-opacity"
        title="Delete"
        aria-label="Delete image"
      >
        <Trash2 size={13} className="text-muted-foreground hover:text-red-500" />
      </button>
    </button>
  );
}
