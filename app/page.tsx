'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import Sidebar from '@/components/sidebar/Sidebar';
import NoteEditor from '@/components/editor/Editor';
import type { NoteEditorHandle } from '@/components/editor/Editor';
import SummaryPanel from '@/components/related/RelatedPanel';
import SearchDialog from '@/components/search/SearchDialog';
import DocumentViewer from '@/components/viewer/DocumentViewer';
import StartupScreen from '@/components/startup/StartupScreen';
import { Menu, Lightbulb } from 'lucide-react';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import type { Note, Document, UploadedImage } from '@/lib/types';
import { debounce } from '@/lib/utils';

type ViewState =
  | { mode: 'note' }
  | { mode: 'document'; id: string }
  | { mode: 'image'; id: string };

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);
  return isMobile;
}

export default function HomePage() {
  const [mounted, setMounted] = useState(false);
  const [showStartup, setShowStartup] = useState(true);
  const [notes, setNotes] = useState<Note[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [images, setImages] = useState<UploadedImage[]>([]);
  const [deletedNotes, setDeletedNotes] = useState<Note[]>([]);
  const [selectedNote, setSelectedNote] = useState<Note | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [viewState, setViewState] = useState<ViewState>({ mode: 'note' });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [insightCollapsed, setInsightCollapsed] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [uploadingDoc, setUploadingDoc] = useState(false);
  const [uploadingImg, setUploadingImg] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [mobilePanelOpen, setMobilePanelOpen] = useState(false);
  const isMobile = useIsMobile();
  const isSaving = useRef(false);
  const editorRef = useRef<NoteEditorHandle>(null);

  useEffect(() => setMounted(true), []);

  const fetchAll = useCallback(async () => {
    const [notesRes, docsRes, imgsRes] = await Promise.all([
      fetch('/api/notes').then(r => r.json()).catch(() => []),
      fetch('/api/documents').then(r => r.json()).catch(() => []),
      fetch('/api/images').then(r => r.json()).catch(() => []),
    ]);
    setNotes(notesRes);
    setDocuments(docsRes);
    setImages(imgsRes);
  }, []);

  const fetchDeletedNotes = useCallback(async () => {
    try {
      const res = await fetch('/api/notes/deleted');
      const data = await res.json();
      setDeletedNotes(data);
    } catch {}
  }, []);

  useEffect(() => {
    fetchAll();
    fetchDeletedNotes();

    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null;
    if (savedTheme) setTheme(savedTheme);
  }, [fetchAll, fetchDeletedNotes]);

  const selectedNoteRef = useRef(selectedNote);
  useEffect(() => { selectedNoteRef.current = selectedNote; }, [selectedNote]);

  const titleRef = useRef(title);
  const contentRef = useRef(content);
  useEffect(() => { titleRef.current = title; }, [title]);

  const saveNote = useCallback(async (t: string, c: string) => {
    const note = selectedNoteRef.current;
    if (!note || isSaving.current) return;
    contentRef.current = c;
    isSaving.current = true;
    try {
      const res = await fetch(`/api/notes/${note.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: t, content: c }),
      });
      const updated = await res.json();
      setNotes(prev => prev.map(n => n.id === updated.id ? updated : n));
      setSelectedNote(updated);
      setContent(c);
    } catch (err) {
      console.error('Failed to save note:', err);
    }
    isSaving.current = false;
  }, []);

  const debouncedSave = useMemo(
    () => debounce(() => {
      const c = editorRef.current?.getHTML() ?? contentRef.current;
      saveNote(titleRef.current, c);
    }, 1000),
    [saveNote],
  );

  const handleTitleChange = useCallback((t: string) => {
    setTitle(t);
    titleRef.current = t;
    debouncedSave();
  }, [debouncedSave]);

  const handleContentDirty = useCallback(() => {
    debouncedSave();
  }, [debouncedSave]);

  const selectNote = useCallback((note: Note) => {
    debouncedSave.cancel();
    if (selectedNoteRef.current && selectedNoteRef.current.id !== note.id && editorRef.current) {
      const html = editorRef.current.getHTML();
      if (html !== contentRef.current) {
        const prevId = selectedNoteRef.current.id;
        const prevTitle = titleRef.current;
        fetch(`/api/notes/${prevId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: prevTitle, content: html }),
        }).then(r => r.json()).then(updated => {
          setNotes(prev => prev.map(n => n.id === updated.id ? updated : n));
        }).catch(() => {});
      }
    }
    selectedNoteRef.current = note;
    titleRef.current = note.title;
    contentRef.current = note.content;
    setSelectedNote(note);
    setTitle(note.title);
    setContent(note.content);
    setViewState({ mode: 'note' });
    setMobileSidebarOpen(false);
  }, [debouncedSave]);

  const createNote = useCallback(async () => {
    debouncedSave.cancel();

    const prevNote = selectedNoteRef.current;
    if (prevNote && editorRef.current) {
      const html = editorRef.current.getHTML();
      if (html !== contentRef.current) {
        isSaving.current = true;
        try {
          await fetch(`/api/notes/${prevNote.id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: titleRef.current, content: html }),
          });
        } catch {}
        isSaving.current = false;
      }
    }

    selectedNoteRef.current = null;
    titleRef.current = 'Untitled';
    contentRef.current = '';

    try {
      const res = await fetch('/api/notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'Untitled', content: '' }),
      });
      const note = await res.json();
      selectedNoteRef.current = note;
      setNotes(prev => [note, ...prev]);
      setSelectedNote(note);
      setTitle('Untitled');
      setContent('');
      setViewState({ mode: 'note' });
      setMobileSidebarOpen(false);
    } catch (err) {
      console.error('Failed to create note:', err);
    }
  }, [debouncedSave]);

  const deleteNote = useCallback(async (id: string) => {
    try {
      await fetch(`/api/notes/${id}`, { method: 'DELETE' });
      setNotes(prev => prev.filter(n => n.id !== id));
      if (selectedNote?.id === id) {
        setSelectedNote(null);
        setTitle('');
        setContent('');
      }
      fetchDeletedNotes();
    } catch (err) {
      console.error('Failed to delete note:', err);
    }
  }, [selectedNote, fetchDeletedNotes]);

  const restoreNote = useCallback(async (id: string) => {
    try {
      await fetch(`/api/notes/deleted/${id}`, { method: 'POST' });
      fetchAll();
      fetchDeletedNotes();
    } catch {}
  }, [fetchAll, fetchDeletedNotes]);

  const permanentDeleteNote = useCallback(async (id: string) => {
    try {
      await fetch(`/api/notes/deleted/${id}`, { method: 'DELETE' });
      fetchDeletedNotes();
    } catch {}
  }, [fetchDeletedNotes]);

  const uploadDocument = useCallback(async (file: File) => {
    setUploadingDoc(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', 'document');
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.document) {
        setDocuments(prev => [data.document, ...prev]);
        setViewState({ mode: 'document', id: data.document.id });
        pollDocumentStatus(data.document.id);
      }
    } catch (err) {
      console.error('Upload document error:', err);
    }
    setUploadingDoc(false);
  }, []);

  const uploadImage = useCallback(async (file: File) => {
    setUploadingImg(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', 'image');
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.image) {
        setImages(prev => [data.image, ...prev]);
        setViewState({ mode: 'image', id: data.image.id });
        pollImageStatus(data.image.id);
      }
    } catch (err) {
      console.error('Upload image error:', err);
    }
    setUploadingImg(false);
  }, []);

  const pollIntervalsRef = useRef<Set<ReturnType<typeof setInterval>>>(new Set());

  useEffect(() => {
    return () => {
      pollIntervalsRef.current.forEach(clearInterval);
    };
  }, []);

  const pollDocumentStatus = useCallback((id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/documents/${id}`);
        const doc = await res.json();
        if (doc.status !== 'processing') {
          clearInterval(interval);
          pollIntervalsRef.current.delete(interval);
          setDocuments(prev => prev.map(d => d.id === id ? doc : d));
        }
      } catch {
        clearInterval(interval);
        pollIntervalsRef.current.delete(interval);
      }
    }, 2000);
    pollIntervalsRef.current.add(interval);
    setTimeout(() => {
      clearInterval(interval);
      pollIntervalsRef.current.delete(interval);
    }, 120000);
  }, []);

  const pollImageStatus = useCallback((id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/images/${id}`);
        const img = await res.json();
        if (img.status !== 'processing') {
          clearInterval(interval);
          pollIntervalsRef.current.delete(interval);
          setImages(prev => prev.map(i => i.id === id ? img : i));
        }
      } catch {
        clearInterval(interval);
        pollIntervalsRef.current.delete(interval);
      }
    }, 2000);
    pollIntervalsRef.current.add(interval);
    setTimeout(() => {
      clearInterval(interval);
      pollIntervalsRef.current.delete(interval);
    }, 120000);
  }, []);

  const deleteDocument = useCallback(async (id: string) => {
    await fetch(`/api/documents/${id}`, { method: 'DELETE' });
    setDocuments(prev => prev.filter(d => d.id !== id));
    if (viewState.mode === 'document' && viewState.id === id) setViewState({ mode: 'note' });
  }, [viewState]);

  const deleteImage = useCallback(async (id: string) => {
    await fetch(`/api/images/${id}`, { method: 'DELETE' });
    setImages(prev => prev.filter(i => i.id !== id));
    if (viewState.mode === 'image' && viewState.id === id) setViewState({ mode: 'note' });
  }, [viewState]);

  const viewDocument = useCallback((doc: Document) => {
    setViewState({ mode: 'document', id: doc.id });
    setSelectedNote(null);
    setMobileSidebarOpen(false);
  }, []);

  const viewImage = useCallback((img: UploadedImage) => {
    setViewState({ mode: 'image', id: img.id });
    setSelectedNote(null);
    setMobileSidebarOpen(false);
  }, []);

  const handleSearchViewDoc = useCallback((id: string) => {
    setViewState({ mode: 'document', id });
    setSelectedNote(null);
  }, []);

  const handleSearchViewImg = useCallback((id: string) => {
    setViewState({ mode: 'image', id });
    setSelectedNote(null);
  }, []);

  const handleThemeChange = useCallback((newTheme: 'light' | 'dark') => {
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    document.documentElement.classList.remove('light', 'dark');
    document.documentElement.classList.add(newTheme);
    fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ theme: newTheme }),
    }).catch(() => {});
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen(true);
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
        e.preventDefault();
        createNote();
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        debouncedSave.cancel();
        const c = editorRef.current?.getHTML() ?? contentRef.current;
        saveNote(titleRef.current, c);
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'b' && e.shiftKey) {
        e.preventDefault();
        setSidebarCollapsed(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [createNote, saveNote, debouncedSave]);

  const handleTopicClick = useCallback((topic: string) => {
    const keyword = topic.toLowerCase();

    window.getSelection()?.removeAllRanges();

    const mainContent = document.querySelector('.tiptap') || document.querySelector('.prose');
    if (!mainContent) return;

    const walker = document.createTreeWalker(mainContent, NodeFilter.SHOW_TEXT);
    let node: Node | null;
    while ((node = walker.nextNode())) {
      const text = node.textContent?.toLowerCase() || '';
      const idx = text.indexOf(keyword);
      if (idx !== -1) {
        const range = document.createRange();
        range.setStart(node, idx);
        range.setEnd(node, idx + keyword.length);

        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);

        const el = node.parentElement;
        el?.scrollIntoView({ behavior: 'smooth', block: 'center' });

        if (el) {
          el.classList.add('topic-highlight');
          setTimeout(() => el.classList.remove('topic-highlight'), 3000);
        }
        break;
      }
    }
  }, []);

  const selectedId = viewState.mode === 'note' ? selectedNote?.id ?? null
    : viewState.mode === 'document' ? viewState.id
    : viewState.mode === 'image' ? viewState.id
    : null;

  const summaryId = viewState.mode === 'note' ? selectedNote?.id ?? null
    : viewState.mode === 'document' || viewState.mode === 'image' ? viewState.id
    : null;
  const summaryContent = viewState.mode === 'note' ? content
    : viewState.mode === 'document' ? documents.find(d => d.id === viewState.id)?.plain_text || ''
    : viewState.mode === 'image' ? images.find(i => i.id === viewState.id)?.ocr_text || ''
    : '';

  if (!mounted) return null;

  return (
    <ErrorBoundary>
    {showStartup && <StartupScreen onReady={() => setShowStartup(false)} />}
    <div className="flex h-screen overflow-hidden relative">
      {isMobile && (
        <div className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-3 py-2 bg-card border-b border-border">
          <button
            onClick={() => { setMobileSidebarOpen(true); setMobilePanelOpen(false); }}
            className="p-2 rounded-md hover:bg-accent text-muted-foreground"
            aria-label="Open sidebar"
          >
            <Menu size={20} />
          </button>
          <h1 className="text-sm font-bold text-foreground">Epito</h1>
          <button
            onClick={() => { setMobilePanelOpen(true); setMobileSidebarOpen(false); }}
            className="p-2 rounded-md hover:bg-accent text-muted-foreground"
            aria-label="Open insights panel"
          >
            <Lightbulb size={20} />
          </button>
        </div>
      )}

      {isMobile ? (
        <>
          {mobileSidebarOpen && (
            <div className="fixed inset-0 z-50 bg-black/50" onClick={() => setMobileSidebarOpen(false)} />
          )}
          <div className={`fixed inset-y-0 left-0 z-50 w-[85vw] max-w-72 transition-transform duration-200 ease-out ${
            mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}>
            <Sidebar
              notes={notes}
              documents={documents}
              images={images}
              selectedId={selectedId}
              collapsed={false}
              onToggleCollapse={() => setMobileSidebarOpen(false)}
              onSelect={selectNote}
              onCreate={createNote}
              onDelete={deleteNote}
              onSearchOpen={() => { setSearchOpen(true); setMobileSidebarOpen(false); }}
              onUploadDocument={uploadDocument}
              onUploadImage={uploadImage}
              onViewDocument={viewDocument}
              onViewImage={viewImage}
              onDeleteDocument={deleteDocument}
              onDeleteImage={deleteImage}
              deletedNotes={deletedNotes}
              onRestoreNote={restoreNote}
              onPermanentDelete={permanentDeleteNote}
              theme={theme}
              onThemeChange={handleThemeChange}
              uploadingDoc={uploadingDoc}
              uploadingImg={uploadingImg}
            />
          </div>
        </>
      ) : (
        <Sidebar
          notes={notes}
          documents={documents}
          images={images}
          selectedId={selectedId}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(prev => !prev)}
          onSelect={selectNote}
          onCreate={createNote}
          onDelete={deleteNote}
          onSearchOpen={() => setSearchOpen(true)}
          onUploadDocument={uploadDocument}
          onUploadImage={uploadImage}
          onViewDocument={viewDocument}
          onViewImage={viewImage}
          onDeleteDocument={deleteDocument}
          onDeleteImage={deleteImage}
          deletedNotes={deletedNotes}
          onRestoreNote={restoreNote}
          onPermanentDelete={permanentDeleteNote}
          theme={theme}
          onThemeChange={handleThemeChange}
          uploadingDoc={uploadingDoc}
          uploadingImg={uploadingImg}
        />
      )}

      <div className={`flex-1 flex flex-col overflow-hidden ${isMobile ? 'pt-11' : ''}`}>
        {viewState.mode === 'note' ? (
          <NoteEditor
            ref={editorRef}
            noteId={selectedNote?.id ?? null}
            title={title}
            content={content}
            createdAt={selectedNote?.created_at}
            updatedAt={selectedNote?.updated_at}
            onTitleChange={handleTitleChange}
            onContentDirty={handleContentDirty}
            isMobile={isMobile}
          />
        ) : (
          <DocumentViewer
            type={viewState.mode === 'document' ? 'document' : 'image'}
            id={viewState.id}
            onClose={() => setViewState({ mode: 'note' })}
            isMobile={isMobile}
            onTextChange={(docId, docType, text) => {
              if (docType === 'document') {
                setDocuments(prev => prev.map(d => d.id === docId ? { ...d, plain_text: text } : d));
              } else {
                setImages(prev => prev.map(i => i.id === docId ? { ...i, ocr_text: text } : i));
              }
            }}
          />
        )}
      </div>

      {isMobile ? (
        <>
          {mobilePanelOpen && (
            <div className="fixed inset-0 z-50 bg-black/50" onClick={() => setMobilePanelOpen(false)} />
          )}
          <div className={`fixed inset-y-0 right-0 z-50 w-full max-w-sm transition-transform duration-200 ease-out ${
            mobilePanelOpen ? 'translate-x-0' : 'translate-x-full'
          }`}>
            <SummaryPanel
              noteId={summaryId}
              noteContent={summaryContent}
              onTopicClick={(topic) => { handleTopicClick(topic); setMobilePanelOpen(false); }}
              isMobile={isMobile}
              onClose={() => setMobilePanelOpen(false)}
            />
          </div>
        </>
      ) : (
        <SummaryPanel
          noteId={summaryId}
          noteContent={summaryContent}
          onTopicClick={handleTopicClick}
          collapsed={insightCollapsed}
          onToggleCollapse={() => setInsightCollapsed(prev => !prev)}
        />
      )}

      <SearchDialog
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
        onSelect={selectNote}
        onViewDocument={handleSearchViewDoc}
        onViewImage={handleSearchViewImg}
      />
    </div>
    </ErrorBoundary>
  );
}
