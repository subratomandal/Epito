import { NextRequest } from 'next/server';
import {
  summarizeTextStream, summarizeChunksStream,
  explainTextStream,
  chatWithRAGStream, classifyQuery, getGreetingResponse,
  summarizeSectionStream, mergeSectionsStream,
  explainSectionStream,
  cleanInputText,
} from '@/lib/ai/llm';
import { retrieveChunksForSummarization, contextualRetrieveForChat } from '@/lib/ai/pipeline';
import { canAcceptTask, taskStarted, taskCompleted, isShuttingDown } from '@/lib/lifecycle';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  if (isShuttingDown()) {
    return new Response(JSON.stringify({ error: 'Application is shutting down' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    });
  }
  if (!canAcceptTask()) {
    return new Response(JSON.stringify({ error: 'Too many concurrent AI tasks. Please wait.' }), {
      status: 429,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: 'Invalid JSON' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }
  const { action, text, sourceId, chatMessage, chatHistory, sectionText, sectionIndex, totalSections, previousPoints, previousContext, sectionSummaries } = body;

  if (!text?.trim() && action !== 'summarize-section' && action !== 'merge-sections' && action !== 'explain-section') {
    return new Response(JSON.stringify({ error: 'Text required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const cleaned = text ? cleanInputText(text) : '';
  const encoder = new TextEncoder();

  taskStarted();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        if (action === 'summarize') {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ progress: 'Retrieving relevant content...' })}\n\n`));
          const chunks = await retrieveChunksForSummarization(sourceId || null, cleaned, 10);

          if (chunks.length > 0) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ progress: 'Analyzing content and extracting insights...' })}\n\n`));

            for await (const chunk of summarizeChunksStream(chunks)) {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: chunk })}\n\n`));
            }
          } else {
            for await (const chunk of summarizeTextStream(cleaned)) {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: chunk })}\n\n`));
            }
          }
        } else if (action === 'explain') {
          for await (const chunk of explainTextStream(cleaned)) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: chunk })}\n\n`));
          }
        } else if (action === 'chat') {
          if (!chatMessage) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: 'chatMessage required' })}\n\n`));
            controller.close();
            return;
          }

          const queryType = classifyQuery(chatMessage);

          if (queryType === 'greeting' || queryType === 'casual') {
            const greeting = getGreetingResponse();
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: greeting })}\n\n`));
          } else {
            const retrieval = await contextualRetrieveForChat(sourceId || null, chatMessage, 5);

            if (retrieval.contexts.length > 0) {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                progress: retrieval.method === 'contextual'
                  ? `Found ${retrieval.sources.length} relevant section${retrieval.sources.length > 1 ? 's' : ''} in document...`
                  : 'Searching document...',
              })}\n\n`));

              controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                retrieval: {
                  method: retrieval.method,
                  sources: retrieval.sources,
                },
              })}\n\n`));

              for await (const chunk of chatWithRAGStream(retrieval.contexts, chatMessage, chatHistory || [])) {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: chunk })}\n\n`));
              }
            } else {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: 'No relevant content found in the document to answer your question. Try rephrasing or ensure the document has been processed.' })}\n\n`));
            }
          }
        } else if (action === 'summarize-section') {
          if (!sectionText) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: 'sectionText required' })}\n\n`));
            controller.close();
            return;
          }
          for await (const chunk of summarizeSectionStream(sectionText, sectionIndex || 0, totalSections || 1, previousPoints || '')) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: chunk })}\n\n`));
          }
        } else if (action === 'explain-section') {
          if (!sectionText) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: 'sectionText required' })}\n\n`));
            controller.close();
            return;
          }

          let surroundingContext = '';
          if (cleaned) {
            const words = cleaned.split(/\s+/).filter((w: string) => w.length > 0);
            const sectionWordCount = 100;
            const sectionStartWord = (sectionIndex || 0) * sectionWordCount;
            const ctxStart = Math.max(0, sectionStartWord - 150);
            const ctxEnd = Math.min(words.length, sectionStartWord + sectionWordCount + 150);
            const before = words.slice(ctxStart, sectionStartWord).join(' ');
            const after = words.slice(Math.min(words.length, sectionStartWord + sectionWordCount), ctxEnd).join(' ');
            surroundingContext = [before, after].filter(Boolean).join(' ... ');
          }

          for await (const chunk of explainSectionStream(sectionText, sectionIndex || 0, totalSections || 1, previousContext || '', surroundingContext)) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: chunk })}\n\n`));
          }
        } else if (action === 'merge-sections') {
          if (!sectionSummaries?.length) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: 'sectionSummaries required' })}\n\n`));
            controller.close();
            return;
          }
          for await (const chunk of mergeSectionsStream(sectionSummaries)) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: chunk })}\n\n`));
          }
        }

        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
        taskCompleted();
      } catch (err) {
        console.error('[API] Stream error:', err);
        const msg = err instanceof Error ? err.message : 'Generation failed';
        const errorText = msg.includes('ECONNREFUSED') || msg.includes('fetch failed')
          ? 'Cannot connect to AI engine. It may still be loading.'
          : msg;
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: errorText })}\n\n`));
        controller.close();
        taskCompleted();
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
