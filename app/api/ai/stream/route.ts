import { NextRequest } from 'next/server';
import {
  splitNoteIntoBlocks,
  summarizeBatchStream,
  synthesizeFinalStream,
  directSummarizeStream,
  msrChatStream,
  cleanInputText,
} from '@/model/llm';
import { splitInto100WordChunks, explainSingleChunk } from '@/model/explain';
import { canAcceptTask, taskStarted, taskCompleted, isShuttingDown } from '@/inference/lifecycle';

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
  const { action, text, sourceId, chatMessage, chatHistory,
    blocks, startIdx, batchSize, previousBlockSummaries, allBlockSummaries } = body;

  if (!text?.trim() && action !== 'explain-section' && action !== 'explain-chunk' && action !== 'summarize-batch' && action !== 'summarize-final') {
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
          const wc = cleaned.split(/\s+/).length;
          if (wc <= 800) {
            for await (const event of directSummarizeStream(cleaned)) {
              if (event.type === 'progress') {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify({ progress: event.message })}\n\n`));
              } else if (event.type === 'final') {
                controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: event.text })}\n\n`));
              }
            }
          } else {
            const noteBlocks = splitNoteIntoBlocks(cleaned);
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ blocks: noteBlocks })}\n\n`));
          }
        } else if (action === 'summarize-batch') {
          const gen = summarizeBatchStream(
            blocks || [],
            startIdx || 0,
            batchSize || 3,
            previousBlockSummaries || [],
          );
          for await (const event of gen) {
            if (event.type === 'progress') {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ progress: event.message })}\n\n`));
            } else if (event.type === 'blockStream') {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                blockStream: event.text,
                blockIndex: event.index,
                totalBlocks: event.total,
              })}\n\n`));
            } else if (event.type === 'blockDone') {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                blockDone: event.summary,
                blockIndex: event.index,
                totalBlocks: event.total,
              })}\n\n`));
            }
          }
        } else if (action === 'summarize-final') {
          for await (const event of synthesizeFinalStream(allBlockSummaries || [])) {
            if (event.type === 'progress') {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ progress: event.message })}\n\n`));
            } else if (event.type === 'final') {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: event.text })}\n\n`));
            }
          }
        } else if (action === 'explain') {
          const chunks = splitInto100WordChunks(cleaned);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            explainChunks: chunks.map(c => c.text),
          })}\n\n`));
        } else if (action === 'explain-chunk') {
          const chunkText = body.chunkText || '';
          const chunkIndex = body.chunkIndex || 0;
          const totalChunks = body.totalChunks || 1;

          for await (const event of explainSingleChunk(chunkText, chunkIndex, totalChunks)) {
            if (event.type === 'progress') {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ progress: event.message })}\n\n`));
            } else if (event.type === 'segmentStream') {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: event.text })}\n\n`));
            } else if (event.type === 'segmentDone') {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ explainDone: event.explanation })}\n\n`));
            }
          }
        } else if (action === 'chat') {
          if (!chatMessage) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: 'chatMessage required' })}\n\n`));
            controller.close();
            return;
          }

          for await (const chunk of msrChatStream(sourceId || null, chatMessage, chatHistory || [])) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: chunk })}\n\n`));
          }
        }

        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
        taskCompleted();
      } catch (err) {
        console.error('[API] Stream error:', err);
        const msg = err instanceof Error ? err.message : 'Generation failed';
        let errorText: string;
        if (msg.includes('ECONNREFUSED') || msg.includes('fetch failed')) {
          errorText = 'Cannot connect to AI engine. It may still be loading.';
        } else if (msg.includes('llama-server failed')) {
          errorText = 'AI engine failed to start. Please restart the application.';
        } else {
          errorText = 'AI processing failed. Please try again.';
        }
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
