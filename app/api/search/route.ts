import { NextRequest, NextResponse } from 'next/server';
import { semanticSearch } from '@/lib/ai/pipeline';
import * as db from '@/lib/database';
import type { Note } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }
  const { query, topK: rawTopK = 20 } = body;
  const topK = Math.min(Math.max(1, Number(rawTopK) || 20), 100);

  if (!query?.trim()) {
    return NextResponse.json({ error: 'Query required' }, { status: 400 });
  }

  try {
    const queryLower = query.toLowerCase();

    const textNotes = db.searchNotesByText(query);
    const textDocs = db.searchDocumentsByText(query);
    const textImages = db.searchImagesByText(query);

    const textIds = new Set([
      ...textNotes.map(n => n.id),
      ...textDocs.map(d => d.id),
      ...textImages.map(i => i.id),
    ]);

    const semanticResults = await semanticSearch(query, topK);
    const filteredSemantic = semanticResults.filter(r => {
      if (textIds.has(r.note.id)) return true;
      const title = r.note.title?.toLowerCase() || '';
      const content = r.chunk.content?.toLowerCase() || '';
      const plainText = r.note.plain_text?.toLowerCase() || '';
      return title.includes(queryLower) || content.includes(queryLower) || plainText.includes(queryLower);
    });

    const seenIds = new Set(filteredSemantic.map(r => r.note.id));
    const fallback = [
      ...textNotes.filter(n => !seenIds.has(n.id)).map(note => ({
        note,
        source_type: 'note' as const,
        chunk: { id: '', note_id: note.id, source_type: 'note' as const, content: note.plain_text.slice(0, 300), start_offset: 0, end_offset: 300 },
        score: 0.15,
        matchedTopics: [],
      })),
      ...textDocs.filter(d => !seenIds.has(d.id)).map(doc => ({
        note: { id: doc.id, title: doc.filename, content: '', plain_text: doc.plain_text, folder: 'Documents', tags: doc.tags, created_at: doc.created_at, updated_at: doc.updated_at, chunk_count: doc.chunk_count } as Note,
        document: doc,
        source_type: 'document' as const,
        chunk: { id: '', note_id: doc.id, source_type: 'document' as const, content: doc.plain_text.slice(0, 300), start_offset: 0, end_offset: 300 },
        score: 0.15,
        matchedTopics: [],
      })),
      ...textImages.filter(i => !seenIds.has(i.id)).map(img => ({
        note: { id: img.id, title: img.filename, content: '', plain_text: img.ocr_text, folder: 'Images', tags: img.tags, created_at: img.created_at, updated_at: img.updated_at, chunk_count: img.chunk_count } as Note,
        image: img,
        source_type: 'image' as const,
        chunk: { id: '', note_id: img.id, source_type: 'image' as const, content: img.ocr_text.slice(0, 300), start_offset: 0, end_offset: 300 },
        score: 0.15,
        matchedTopics: [],
      })),
    ];

    const combined = [...filteredSemantic, ...fallback];
    return NextResponse.json(combined.slice(0, topK));
  } catch (err) {
    console.error('[API] Search error:', err);
    const results = db.searchNotesByText(query);
    return NextResponse.json(
      results.map(note => ({
        note,
        source_type: 'note',
        chunk: { id: '', note_id: note.id, source_type: 'note', content: note.plain_text.slice(0, 300), start_offset: 0, end_offset: 300 },
        score: 0.1,
        matchedTopics: [],
      }))
    );
  }
}
