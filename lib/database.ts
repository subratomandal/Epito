import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { randomUUID } from 'crypto';
import type { Note, NoteChunk, Topic, NoteLink, Document, UploadedImage, SourceType } from './types';
import { encrypt, decrypt } from './encryption';

const DATA_DIR = process.env.EPITO_DATA_DIR || path.resolve(process.cwd(), 'data');
fs.mkdirSync(DATA_DIR, { recursive: true });

let _db: Database.Database | null = null;

function getDb(): Database.Database {
  if (!_db) {
    const db = new Database(path.join(DATA_DIR, 'notes.db'));
    db.pragma('journal_mode = WAL');
    db.pragma('foreign_keys = ON');
    db.pragma('synchronous = NORMAL');
    // Checkpoint WAL on open: recovers any leftover WAL from a previous
    // force-kill (e.g. taskkill /F on Windows) and prevents WAL bloat.
    try { db.pragma('wal_checkpoint(TRUNCATE)'); } catch {}
    initSchema(db);
    _db = db;
  }
  return _db;
}

function initSchema(db: Database.Database) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS notes (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL DEFAULT 'Untitled',
      content TEXT NOT NULL DEFAULT '',
      plain_text TEXT NOT NULL DEFAULT '',
      folder TEXT NOT NULL DEFAULT '',
      tags TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      chunk_count INTEGER NOT NULL DEFAULT 0,
      deleted_at TEXT DEFAULT NULL
    );

    CREATE TABLE IF NOT EXISTS documents (
      id TEXT PRIMARY KEY,
      filename TEXT NOT NULL,
      file_type TEXT NOT NULL DEFAULT 'pdf',
      file_path TEXT NOT NULL DEFAULT '',
      file_size INTEGER NOT NULL DEFAULT 0,
      plain_text TEXT NOT NULL DEFAULT '',
      page_count INTEGER NOT NULL DEFAULT 0,
      tags TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      chunk_count INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'processing'
    );

    CREATE TABLE IF NOT EXISTS images (
      id TEXT PRIMARY KEY,
      filename TEXT NOT NULL,
      file_path TEXT NOT NULL DEFAULT '',
      file_size INTEGER NOT NULL DEFAULT 0,
      mime_type TEXT NOT NULL DEFAULT 'image/png',
      ocr_text TEXT NOT NULL DEFAULT '',
      width INTEGER NOT NULL DEFAULT 0,
      height INTEGER NOT NULL DEFAULT 0,
      tags TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      chunk_count INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'processing'
    );

    CREATE TABLE IF NOT EXISTS note_chunks (
      id TEXT PRIMARY KEY,
      note_id TEXT NOT NULL,
      content TEXT NOT NULL,
      start_offset INTEGER NOT NULL DEFAULT 0,
      end_offset INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS embeddings (
      chunk_id TEXT PRIMARY KEY,
      vector BLOB NOT NULL,
      FOREIGN KEY (chunk_id) REFERENCES note_chunks(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS note_embeddings (
      note_id TEXT PRIMARY KEY,
      vector BLOB NOT NULL
    );

    CREATE TABLE IF NOT EXISTS topics (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      keywords TEXT NOT NULL DEFAULT '[]',
      note_ids TEXT NOT NULL DEFAULT '[]',
      frequency INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS links (
      id TEXT PRIMARY KEY,
      source_note_id TEXT NOT NULL,
      target_note_id TEXT NOT NULL,
      weight REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS tags (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL UNIQUE,
      color TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS note_tags (
      note_id TEXT NOT NULL,
      tag_id TEXT NOT NULL,
      PRIMARY KEY (note_id, tag_id),
      FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE,
      FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS note_attachments (
      id TEXT PRIMARY KEY,
      note_id TEXT NOT NULL,
      filename TEXT NOT NULL,
      file_path TEXT NOT NULL,
      file_type TEXT NOT NULL,
      file_size INTEGER NOT NULL DEFAULT 0,
      mime_type TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      encrypted INTEGER NOT NULL DEFAULT 0,
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS chunk_cache (
      id TEXT PRIMARY KEY,
      source_id TEXT NOT NULL,
      chunk_index INTEGER NOT NULL,
      chunk_text TEXT NOT NULL,
      content_hash TEXT NOT NULL,
      summary TEXT DEFAULT NULL,
      explanation TEXT DEFAULT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_chunk_cache_source ON chunk_cache(source_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_cache_source_idx ON chunk_cache(source_id, chunk_index);
    CREATE INDEX IF NOT EXISTS idx_chunks_note ON note_chunks(note_id);
    CREATE INDEX IF NOT EXISTS idx_links_src ON links(source_note_id);
    CREATE INDEX IF NOT EXISTS idx_links_tgt ON links(target_note_id);
    CREATE INDEX IF NOT EXISTS idx_note_tags_note ON note_tags(note_id);
    CREATE INDEX IF NOT EXISTS idx_note_tags_tag ON note_tags(tag_id);
    CREATE INDEX IF NOT EXISTS idx_attachments_note ON note_attachments(note_id);
  `);

  try {
    db.prepare("SELECT source_type FROM note_chunks LIMIT 1").get();
  } catch {
    try { db.exec("ALTER TABLE note_chunks ADD COLUMN source_type TEXT NOT NULL DEFAULT 'note'"); } catch {}
  }

  try {
    db.prepare("SELECT deleted_at FROM notes LIMIT 1").get();
  } catch {
    try { db.exec("ALTER TABLE notes ADD COLUMN deleted_at TEXT DEFAULT NULL"); } catch {}
  }

  try {
    db.exec("CREATE INDEX IF NOT EXISTS idx_chunks_source ON note_chunks(source_type)");
  } catch {}

  migrateTagsToRelational(db);
}

function migrateTagsToRelational(db: Database.Database) {
  const count = (db.prepare('SELECT COUNT(*) as c FROM note_tags').get() as { c: number }).c;
  const noteCount = (db.prepare('SELECT COUNT(*) as c FROM notes').get() as { c: number }).c;
  if (count > 0 || noteCount === 0) return;

  const notes = db.prepare('SELECT id, tags FROM notes').all() as { id: string; tags: string }[];
  const insertTag = db.prepare('INSERT OR IGNORE INTO tags (id, name) VALUES (?, ?)');
  const insertNoteTag = db.prepare('INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)');
  const getTag = db.prepare('SELECT id FROM tags WHERE name = ?');

  const transaction = db.transaction(() => {
    for (const note of notes) {
      const tagNames: string[] = JSON.parse(note.tags || '[]');
      for (const name of tagNames) {
        const trimmed = name.trim();
        if (!trimmed) continue;
        let existing = getTag.get(trimmed) as { id: string } | undefined;
        if (!existing) {
          const tagId = randomUUID();
          insertTag.run(tagId, trimmed);
          existing = { id: tagId };
        }
        insertNoteTag.run(note.id, existing.id);
      }
    }
  });
  transaction();
}

function escapeLike(s: string): string {
  return s.replace(/[%_\\]/g, '\\$&');
}

function toBlob(v: number[]): Buffer {
  return Buffer.from(new Float32Array(v).buffer);
}
function fromBlob(buf: Buffer): number[] {
  return Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
}

export function createNote(title: string, content = '', plainText = '', folder = '', tags: string[] = []): string {
  const db = getDb();
  const id = randomUUID();
  db.prepare(
    `INSERT INTO notes (id, title, content, plain_text, folder, tags) VALUES (?, ?, ?, ?, ?, ?)`
  ).run(id, title, content, plainText, folder, JSON.stringify(tags));

  syncNoteTags(id, tags);

  return id;
}

export function updateNote(id: string, fields: Partial<Pick<Note, 'title' | 'content' | 'plain_text' | 'folder' | 'tags'>>) {
  const db = getDb();
  const sets: string[] = ["updated_at = datetime('now')"];
  const vals: unknown[] = [];

  if (fields.title !== undefined) { sets.push('title = ?'); vals.push(fields.title); }
  if (fields.content !== undefined) { sets.push('content = ?'); vals.push(fields.content); }
  if (fields.plain_text !== undefined) { sets.push('plain_text = ?'); vals.push(fields.plain_text); }
  if (fields.folder !== undefined) { sets.push('folder = ?'); vals.push(fields.folder); }
  if (fields.tags !== undefined) {
    sets.push('tags = ?');
    vals.push(JSON.stringify(fields.tags));
    syncNoteTags(id, fields.tags);
  }

  vals.push(id);
  db.prepare(`UPDATE notes SET ${sets.join(', ')} WHERE id = ?`).run(...vals);
}

export function getNote(id: string): Note | null {
  const db = getDb();
  const row = db.prepare('SELECT * FROM notes WHERE id = ?').get(id) as Record<string, unknown> | undefined;
  return row ? parseNote(row) : null;
}

export function getAllNotes(): Note[] {
  const db = getDb();
  return (db.prepare('SELECT * FROM notes WHERE deleted_at IS NULL ORDER BY updated_at DESC').all() as Record<string, unknown>[]).map(parseNote);
}

export function deleteNote(id: string) {
  getDb().prepare("UPDATE notes SET deleted_at = datetime('now'), updated_at = datetime('now') WHERE id = ?").run(id);
}

export function permanentDeleteNote(id: string) {
  const db = getDb();

  const attachments = db.prepare('SELECT file_path FROM note_attachments WHERE note_id = ?').all(id) as { file_path: string }[];

  const deleteAll = db.transaction(() => {
    db.prepare('DELETE FROM note_tags WHERE note_id = ?').run(id);
    db.prepare('DELETE FROM note_attachments WHERE note_id = ?').run(id);
    db.prepare('DELETE FROM note_chunks WHERE note_id = ?').run(id);
    db.prepare('DELETE FROM note_embeddings WHERE note_id = ?').run(id);
    db.prepare('DELETE FROM chunk_cache WHERE source_id = ?').run(id);
    db.prepare("DELETE FROM chunk_cache WHERE source_id = ? OR source_id = ?").run(`${id}:summary`, `${id}:explain`);
    db.prepare('DELETE FROM links WHERE source_note_id = ? OR target_note_id = ?').run(id, id);
    db.prepare('DELETE FROM notes WHERE id = ?').run(id);
  });
  deleteAll();

  const uploadDir = path.resolve(DATA_DIR, 'uploads');
  for (const att of attachments) {
    if (!att.file_path) continue;
    const filename = path.basename(att.file_path);
    if (filename) {
      try { fs.unlinkSync(path.join(uploadDir, filename)); } catch {}
    }
  }
}

export function restoreNote(id: string) {
  getDb().prepare("UPDATE notes SET deleted_at = NULL, updated_at = datetime('now') WHERE id = ?").run(id);
}

export function getDeletedNotes(): Note[] {
  const db = getDb();
  return (db.prepare("SELECT * FROM notes WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC").all() as Record<string, unknown>[]).map(parseNote);
}

export function purgeOldDeleted(days = 7): number {
  const db = getDb();
  const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
  const toDelete = db.prepare("SELECT id FROM notes WHERE deleted_at IS NOT NULL AND deleted_at < ?").all(cutoff) as { id: string }[];
  for (const { id } of toDelete) {
    permanentDeleteNote(id);
  }
  return toDelete.length;
}

export function searchNotesByText(query: string): Note[] {
  const db = getDb();
  const escaped = escapeLike(query);
  return (db.prepare("SELECT * FROM notes WHERE deleted_at IS NULL AND (plain_text LIKE ? ESCAPE '\\' OR title LIKE ? ESCAPE '\\') ORDER BY updated_at DESC LIMIT 50")
    .all(`%${escaped}%`, `%${escaped}%`) as Record<string, unknown>[]).map(parseNote);
}

function parseNote(row: Record<string, unknown>): Note {
  return {
    id: row.id as string,
    title: row.title as string,
    content: row.content as string,
    plain_text: row.plain_text as string,
    folder: row.folder as string,
    tags: JSON.parse((row.tags as string) || '[]'),
    created_at: row.created_at as string,
    updated_at: row.updated_at as string,
    chunk_count: row.chunk_count as number,
    deleted_at: (row.deleted_at as string) || undefined,
  };
}

export function syncNoteTags(noteId: string, tagNames: string[]) {
  const db = getDb();
  const sync = db.transaction(() => {
    db.prepare('DELETE FROM note_tags WHERE note_id = ?').run(noteId);

    const insertTag = db.prepare('INSERT OR IGNORE INTO tags (id, name) VALUES (?, ?)');
    const getTag = db.prepare('SELECT id FROM tags WHERE name = ?');
    const insertNoteTag = db.prepare('INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)');

    for (const name of tagNames) {
      const trimmed = name.trim();
      if (!trimmed) continue;
      let existing = getTag.get(trimmed) as { id: string } | undefined;
      if (!existing) {
        const tagId = randomUUID();
        insertTag.run(tagId, trimmed);
        existing = { id: tagId };
      }
      insertNoteTag.run(noteId, existing.id);
    }
  });
  sync();
}

export function getAllTagsWithCounts(): { id: string; name: string; count: number }[] {
  const db = getDb();
  return db.prepare(`
    SELECT t.id, t.name, COUNT(nt.note_id) as count
    FROM tags t
    LEFT JOIN note_tags nt ON t.id = nt.tag_id
    GROUP BY t.id, t.name
    ORDER BY count DESC, t.name ASC
  `).all() as { id: string; name: string; count: number }[];
}

export function createAttachment(noteId: string, filename: string, filePath: string, fileType: string, fileSize: number, mimeType: string): string {
  const db = getDb();
  const id = randomUUID();
  db.prepare(
    `INSERT INTO note_attachments (id, note_id, filename, file_path, file_type, file_size, mime_type) VALUES (?, ?, ?, ?, ?, ?, ?)`
  ).run(id, noteId, filename, filePath, fileType, fileSize, mimeType);
  return id;
}

export function getAttachmentsByNote(noteId: string): { id: string; note_id: string; filename: string; file_path: string; file_type: string; file_size: number; mime_type: string; created_at: string }[] {
  return getDb().prepare('SELECT * FROM note_attachments WHERE note_id = ? ORDER BY created_at DESC').all(noteId) as any[];
}

export function getAttachment(id: string): { id: string; note_id: string; filename: string; file_path: string; file_type: string; file_size: number; mime_type: string; created_at: string } | null {
  return getDb().prepare('SELECT * FROM note_attachments WHERE id = ?').get(id) as any || null;
}

export function deleteAttachment(id: string) {
  getDb().prepare('DELETE FROM note_attachments WHERE id = ?').run(id);
}

export function createDocument(fields: {
  filename: string; fileType: string; filePath: string; fileSize: number;
  plainText: string; pageCount: number; tags?: string[];
}): string {
  const db = getDb();
  const id = randomUUID();
  db.prepare(
    `INSERT INTO documents (id, filename, file_type, file_path, file_size, plain_text, page_count, tags, status)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'processing')`
  ).run(id, fields.filename, fields.fileType, fields.filePath, fields.fileSize,
    fields.plainText, fields.pageCount, JSON.stringify(fields.tags || []));
  return id;
}

export function updateDocument(id: string, fields: Partial<{ plain_text: string; page_count: number; chunk_count: number; status: string; tags: string[] }>) {
  const db = getDb();
  const sets: string[] = ["updated_at = datetime('now')"];
  const vals: unknown[] = [];
  if (fields.plain_text !== undefined) { sets.push('plain_text = ?'); vals.push(fields.plain_text); }
  if (fields.page_count !== undefined) { sets.push('page_count = ?'); vals.push(fields.page_count); }
  if (fields.chunk_count !== undefined) { sets.push('chunk_count = ?'); vals.push(fields.chunk_count); }
  if (fields.status !== undefined) { sets.push('status = ?'); vals.push(fields.status); }
  if (fields.tags !== undefined) { sets.push('tags = ?'); vals.push(JSON.stringify(fields.tags)); }
  vals.push(id);
  db.prepare(`UPDATE documents SET ${sets.join(', ')} WHERE id = ?`).run(...vals);
}

export function getDocument(id: string): Document | null {
  const row = getDb().prepare('SELECT * FROM documents WHERE id = ?').get(id) as Record<string, unknown> | undefined;
  return row ? parseDocument(row) : null;
}

export function getAllDocuments(): Document[] {
  return (getDb().prepare('SELECT * FROM documents ORDER BY created_at DESC').all() as Record<string, unknown>[]).map(parseDocument);
}

export function deleteDocument(id: string) {
  const db = getDb();
  const deleteAll = db.transaction(() => {
    db.prepare('DELETE FROM note_chunks WHERE note_id = ? AND source_type = ?').run(id, 'document');
    db.prepare('DELETE FROM note_embeddings WHERE note_id = ?').run(id);
    db.prepare('DELETE FROM chunk_cache WHERE source_id = ?').run(id);
    db.prepare("DELETE FROM chunk_cache WHERE source_id = ? OR source_id = ?").run(`${id}:summary`, `${id}:explain`);
    db.prepare('DELETE FROM documents WHERE id = ?').run(id);
  });
  deleteAll();
}

export function searchDocumentsByText(query: string): Document[] {
  const escaped = escapeLike(query);
  return (getDb().prepare("SELECT * FROM documents WHERE plain_text LIKE ? ESCAPE '\\' OR filename LIKE ? ESCAPE '\\' ORDER BY created_at DESC LIMIT 50")
    .all(`%${escaped}%`, `%${escaped}%`) as Record<string, unknown>[]).map(parseDocument);
}

function parseDocument(row: Record<string, unknown>): Document {
  return {
    id: row.id as string,
    filename: row.filename as string,
    file_type: row.file_type as 'pdf' | 'docx',
    file_path: row.file_path as string,
    file_size: row.file_size as number,
    plain_text: row.plain_text as string,
    page_count: row.page_count as number,
    tags: JSON.parse((row.tags as string) || '[]'),
    created_at: row.created_at as string,
    updated_at: row.updated_at as string,
    chunk_count: row.chunk_count as number,
    status: row.status as 'processing' | 'ready' | 'error',
  };
}

export function createImage(fields: {
  filename: string; filePath: string; fileSize: number; mimeType: string;
  width: number; height: number; tags?: string[];
}): string {
  const db = getDb();
  const id = randomUUID();
  db.prepare(
    `INSERT INTO images (id, filename, file_path, file_size, mime_type, width, height, tags, status)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'processing')`
  ).run(id, fields.filename, fields.filePath, fields.fileSize, fields.mimeType,
    fields.width, fields.height, JSON.stringify(fields.tags || []));
  return id;
}

export function updateImage(id: string, fields: Partial<{ ocr_text: string; chunk_count: number; status: string; tags: string[] }>) {
  const db = getDb();
  const sets: string[] = ["updated_at = datetime('now')"];
  const vals: unknown[] = [];
  if (fields.ocr_text !== undefined) { sets.push('ocr_text = ?'); vals.push(fields.ocr_text); }
  if (fields.chunk_count !== undefined) { sets.push('chunk_count = ?'); vals.push(fields.chunk_count); }
  if (fields.status !== undefined) { sets.push('status = ?'); vals.push(fields.status); }
  if (fields.tags !== undefined) { sets.push('tags = ?'); vals.push(JSON.stringify(fields.tags)); }
  vals.push(id);
  db.prepare(`UPDATE images SET ${sets.join(', ')} WHERE id = ?`).run(...vals);
}

export function getImage(id: string): UploadedImage | null {
  const row = getDb().prepare('SELECT * FROM images WHERE id = ?').get(id) as Record<string, unknown> | undefined;
  return row ? parseImage(row) : null;
}

export function getAllImages(): UploadedImage[] {
  return (getDb().prepare('SELECT * FROM images ORDER BY created_at DESC').all() as Record<string, unknown>[]).map(parseImage);
}

export function deleteImage(id: string) {
  const db = getDb();
  const deleteAll = db.transaction(() => {
    db.prepare('DELETE FROM note_chunks WHERE note_id = ? AND source_type = ?').run(id, 'image');
    db.prepare('DELETE FROM note_embeddings WHERE note_id = ?').run(id);
    db.prepare('DELETE FROM chunk_cache WHERE source_id = ?').run(id);
    db.prepare("DELETE FROM chunk_cache WHERE source_id = ? OR source_id = ?").run(`${id}:summary`, `${id}:explain`);
    db.prepare('DELETE FROM images WHERE id = ?').run(id);
  });
  deleteAll();
}

export function searchImagesByText(query: string): UploadedImage[] {
  const escaped = escapeLike(query);
  return (getDb().prepare("SELECT * FROM images WHERE ocr_text LIKE ? ESCAPE '\\' OR filename LIKE ? ESCAPE '\\' ORDER BY created_at DESC LIMIT 50")
    .all(`%${escaped}%`, `%${escaped}%`) as Record<string, unknown>[]).map(parseImage);
}

function parseImage(row: Record<string, unknown>): UploadedImage {
  return {
    id: row.id as string,
    filename: row.filename as string,
    file_path: row.file_path as string,
    file_size: row.file_size as number,
    mime_type: row.mime_type as string,
    ocr_text: row.ocr_text as string,
    width: row.width as number,
    height: row.height as number,
    tags: JSON.parse((row.tags as string) || '[]'),
    created_at: row.created_at as string,
    updated_at: row.updated_at as string,
    chunk_count: row.chunk_count as number,
    status: row.status as 'processing' | 'ready' | 'error',
  };
}

export function insertChunk(noteId: string, content: string, startOffset: number, endOffset: number, sourceType: SourceType = 'note'): string {
  const id = randomUUID();
  getDb().prepare('INSERT INTO note_chunks (id, note_id, source_type, content, start_offset, end_offset) VALUES (?, ?, ?, ?, ?, ?)')
    .run(id, noteId, sourceType, content, startOffset, endOffset);
  return id;
}

export function getChunksByNote(noteId: string): NoteChunk[] {
  return (getDb().prepare('SELECT * FROM note_chunks WHERE note_id = ? ORDER BY start_offset').all(noteId) as Record<string, unknown>[])
    .map(parseChunk);
}

export function getAllChunks(): NoteChunk[] {
  return (getDb().prepare('SELECT * FROM note_chunks ORDER BY start_offset').all() as Record<string, unknown>[])
    .map(parseChunk);
}

export function getChunksByIds(ids: string[]): NoteChunk[] {
  if (ids.length === 0) return [];
  const placeholders = ids.map(() => '?').join(',');
  return (getDb().prepare(`SELECT * FROM note_chunks WHERE id IN (${placeholders})`).all(...ids) as Record<string, unknown>[])
    .map(parseChunk);
}

export function deleteChunksBySource(sourceId: string, sourceType: SourceType) {
  getDb().prepare('DELETE FROM note_chunks WHERE note_id = ? AND source_type = ?').run(sourceId, sourceType);
}

export function updateNoteChunkCount(noteId: string, count: number) {
  getDb().prepare('UPDATE notes SET chunk_count = ? WHERE id = ?').run(count, noteId);
}

function parseChunk(row: Record<string, unknown>): NoteChunk {
  return {
    id: row.id as string,
    note_id: row.note_id as string,
    source_type: (row.source_type as SourceType) || 'note',
    content: row.content as string,
    start_offset: row.start_offset as number,
    end_offset: row.end_offset as number,
  };
}

export function insertEmbedding(chunkId: string, vector: number[]) {
  getDb().prepare('INSERT OR REPLACE INTO embeddings (chunk_id, vector) VALUES (?, ?)').run(chunkId, toBlob(vector));
}

export function getAllEmbeddings(): { chunkId: string; vector: number[] }[] {
  return (getDb().prepare('SELECT chunk_id, vector FROM embeddings').all() as { chunk_id: string; vector: Buffer }[])
    .map(r => ({ chunkId: r.chunk_id, vector: fromBlob(r.vector) }));
}

export function getEmbeddingsByChunkIds(chunkIds: string[]): { chunkId: string; vector: number[] }[] {
  if (chunkIds.length === 0) return [];
  const placeholders = chunkIds.map(() => '?').join(',');
  return (getDb().prepare(`SELECT chunk_id, vector FROM embeddings WHERE chunk_id IN (${placeholders})`).all(...chunkIds) as { chunk_id: string; vector: Buffer }[])
    .map(r => ({ chunkId: r.chunk_id, vector: fromBlob(r.vector) }));
}

export function insertNoteEmbedding(noteId: string, vector: number[]) {
  getDb().prepare('INSERT OR REPLACE INTO note_embeddings (note_id, vector) VALUES (?, ?)').run(noteId, toBlob(vector));
}

export function getAllNoteEmbeddings(): { noteId: string; vector: number[] }[] {
  return (getDb().prepare('SELECT note_id, vector FROM note_embeddings').all() as { note_id: string; vector: Buffer }[])
    .map(r => ({ noteId: r.note_id, vector: fromBlob(r.vector) }));
}

export function insertTopic(topic: { id: string; name: string; keywords: string[]; noteIds: string[]; frequency: number }) {
  getDb().prepare('INSERT OR REPLACE INTO topics (id, name, keywords, note_ids, frequency) VALUES (?, ?, ?, ?, ?)')
    .run(topic.id, topic.name, JSON.stringify(topic.keywords), JSON.stringify(topic.noteIds), topic.frequency);
}

export function getAllTopics(): Topic[] {
  return (getDb().prepare('SELECT * FROM topics ORDER BY frequency DESC').all() as Record<string, unknown>[]).map(r => ({
    id: r.id as string,
    name: r.name as string,
    keywords: JSON.parse((r.keywords as string) || '[]'),
    note_ids: JSON.parse((r.note_ids as string) || '[]'),
    frequency: r.frequency as number,
  }));
}

export function deleteAllTopics() {
  const db = getDb();
  db.exec('DELETE FROM links');
  db.exec('DELETE FROM topics');
}

export function insertLink(sourceId: string, targetId: string, weight: number) {
  getDb().prepare('INSERT OR REPLACE INTO links (id, source_note_id, target_note_id, weight) VALUES (?, ?, ?, ?)')
    .run(randomUUID(), sourceId, targetId, weight);
}

export function getAllLinks(): NoteLink[] {
  return getDb().prepare('SELECT * FROM links').all() as NoteLink[];
}

export function deleteAllLinks() {
  getDb().exec('DELETE FROM links');
}

export function getSetting(key: string): string | null {
  const db = getDb();
  const row = db.prepare('SELECT value, encrypted FROM settings WHERE key = ?').get(key) as { value: string; encrypted: number } | undefined;
  if (!row) return null;
  if (row.encrypted) {
    try { return decrypt(row.value); } catch { return null; }
  }
  return row.value;
}

export function setSetting(key: string, value: string, isEncrypted = false) {
  const db = getDb();
  const storedValue = isEncrypted ? encrypt(value) : value;
  db.prepare(
    "INSERT OR REPLACE INTO settings (key, value, encrypted, updated_at) VALUES (?, ?, ?, datetime('now'))"
  ).run(key, storedValue, isEncrypted ? 1 : 0);
}

export function getStats() {
  const db = getDb();
  const notes = (db.prepare('SELECT COUNT(*) as c FROM notes').get() as { c: number }).c;
  const chunks = (db.prepare('SELECT COUNT(*) as c FROM note_chunks').get() as { c: number }).c;
  const topics = (db.prepare('SELECT COUNT(*) as c FROM topics').get() as { c: number }).c;
  const embeddings = (db.prepare('SELECT COUNT(*) as c FROM embeddings').get() as { c: number }).c;
  const documents = (db.prepare('SELECT COUNT(*) as c FROM documents').get() as { c: number }).c;
  const images = (db.prepare('SELECT COUNT(*) as c FROM images').get() as { c: number }).c;
  const tags = (db.prepare('SELECT COUNT(*) as c FROM tags').get() as { c: number }).c;
  return { notes, chunks, topics, embeddings, documents, images, tags };
}

export interface CachedChunk {
  id: string;
  source_id: string;
  chunk_index: number;
  chunk_text: string;
  content_hash: string;
  summary: string | null;
  explanation: string | null;
  created_at: string;
  updated_at: string;
}

export function getChunkCache(sourceId: string): CachedChunk[] {
  return getDb().prepare(
    'SELECT * FROM chunk_cache WHERE source_id = ? ORDER BY chunk_index ASC'
  ).all(sourceId) as CachedChunk[];
}

export function upsertChunkCache(
  sourceId: string,
  chunkIndex: number,
  chunkText: string,
  contentHash: string,
): string {
  const db = getDb();
  const existing = db.prepare(
    'SELECT id FROM chunk_cache WHERE source_id = ? AND chunk_index = ?'
  ).get(sourceId, chunkIndex) as { id: string } | undefined;

  if (existing) {
    const old = db.prepare('SELECT content_hash FROM chunk_cache WHERE id = ?').get(existing.id) as { content_hash: string };
    if (old.content_hash !== contentHash) {
      db.prepare(
        'UPDATE chunk_cache SET chunk_text = ?, content_hash = ?, summary = NULL, explanation = NULL, updated_at = datetime("now") WHERE id = ?'
      ).run(chunkText, contentHash, existing.id);
    }
    return existing.id;
  }

  const id = randomUUID();
  db.prepare(
    'INSERT INTO chunk_cache (id, source_id, chunk_index, chunk_text, content_hash) VALUES (?, ?, ?, ?, ?)'
  ).run(id, sourceId, chunkIndex, chunkText, contentHash);
  return id;
}

export function updateChunkCacheSummary(id: string, summary: string) {
  getDb().prepare(
    'UPDATE chunk_cache SET summary = ?, updated_at = datetime("now") WHERE id = ?'
  ).run(summary, id);
}

export function updateChunkCacheExplanation(id: string, explanation: string) {
  getDb().prepare(
    'UPDATE chunk_cache SET explanation = ?, updated_at = datetime("now") WHERE id = ?'
  ).run(explanation, id);
}

export function clearChunkCache(sourceId: string) {
  getDb().prepare('DELETE FROM chunk_cache WHERE source_id = ?').run(sourceId);
}

export function pruneStaleChunkCache(sourceId: string, validCount: number) {
  getDb().prepare(
    'DELETE FROM chunk_cache WHERE source_id = ? AND chunk_index >= ?'
  ).run(sourceId, validCount);
}

export function closeDb() {
  if (_db) {
    _db.close();
    _db = null;
  }
}
