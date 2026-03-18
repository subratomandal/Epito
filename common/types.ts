export type SourceType = 'note' | 'document' | 'image';

export interface Note {
  id: string;
  title: string;
  content: string;
  plain_text: string;
  folder: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  chunk_count: number;
  deleted_at?: string;
}

export interface Document {
  id: string;
  filename: string;
  file_type: 'pdf' | 'docx';
  file_path: string;
  file_size: number;
  plain_text: string;
  page_count: number;
  tags: string[];
  created_at: string;
  updated_at: string;
  chunk_count: number;
  status: 'processing' | 'ready' | 'error';
}

export interface UploadedImage {
  id: string;
  filename: string;
  file_path: string;
  file_size: number;
  mime_type: string;
  ocr_text: string;
  width: number;
  height: number;
  tags: string[];
  created_at: string;
  updated_at: string;
  chunk_count: number;
  status: 'processing' | 'ready' | 'error';
}

export interface NoteChunk {
  id: string;
  note_id: string;
  source_type: SourceType;
  content: string;
  start_offset: number;
  end_offset: number;
}

export interface Topic {
  id: string;
  name: string;
  keywords: string[];
  note_ids: string[];
  frequency: number;
}

export interface NoteLink {
  id: string;
  source_note_id: string;
  target_note_id: string;
  weight: number;
}

export interface SearchResult {
  note: Note;
  document?: Document;
  image?: UploadedImage;
  source_type: SourceType;
  chunk: NoteChunk;
  score: number;
  matchedTopics: Topic[];
}

export interface RelatedNote {
  note: Note;
  score: number;
}

export interface GraphData {
  topics: Topic[];
  edges: NoteLink[];
}

export interface AISummary {
  summary: string;
  keyPoints: string[];
}

export interface AIStatus {
  embeddings: boolean;
  llm: boolean;
  modelName: string | null;
  vectorCount: number;
}

export interface ContextualMatch {
  context: string;
  matchedTerm: string;
  wordOffset: number;
  snippet: string;
}

export interface ChatRetrievalResult {
  contexts: string[];
  sources: ContextualMatch[];
  method: 'contextual' | 'embedding' | 'fulltext';
}
