import { NextRequest, NextResponse } from 'next/server';
import { summarize, retrieveChunksForSummarization } from '@/lib/ai/pipeline';
import {
  summarizeText, summarizeChunks, explainText,
  msrChat,
  getModelStatus, cleanInputText, chunkForSummarization, chunkForExplain,
} from '@/lib/ai/llm';
import { canAcceptTask, taskStarted, taskCompleted, isShuttingDown } from '@/lib/lifecycle';

export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json(getModelStatus());
}

export async function POST(request: NextRequest) {
  if (isShuttingDown()) {
    return NextResponse.json({ error: 'Application is shutting down' }, { status: 503 });
  }
  if (!canAcceptTask()) {
    return NextResponse.json({ error: 'Too many concurrent AI tasks. Please wait.' }, { status: 429 });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }
  const { action, text, query, sourceId, useExtractive, chatMessage, chatHistory } = body;

  if (!text && !query) {
    return NextResponse.json({ error: 'Text or query required' }, { status: 400 });
  }

  taskStarted();
  try {
    switch (action) {
      case 'summarize':
      case 'summarize-document': {
        if (!useExtractive) {
          const cleaned = cleanInputText(text);

          const chunks = await retrieveChunksForSummarization(sourceId || null, cleaned, 10);

          if (chunks.length > 0) {
            const result = await summarizeChunks(chunks);
            if (result) {
              return NextResponse.json({ ...result, source: 'llm' });
            }
          }

          const directResult = await summarizeText(cleaned);
          if (directResult) {
            return NextResponse.json({ ...directResult, source: 'llm' });
          }
        }
        const sentenceCount = action === 'summarize-document' ? 6 : 4;
        const result = summarize(text, sentenceCount);
        return NextResponse.json({ ...result, source: 'extractive' });
      }

      case 'explain': {
        const cleaned = cleanInputText(text);

        const inputSentences = cleaned
          .split(/(?<=[.!?])\s+/)
          .filter((s: string) => s.trim().length > 10)
          .slice(0, 15);

        if (!useExtractive) {
          const result = await explainText(cleaned);
          if (result) {
            const parsed = parseExplainResponse(result.explanation, inputSentences);
            return NextResponse.json({
              explanation: result.explanation,
              sentences: parsed,
              source: 'llm',
            });
          }
        }

        const fallbackSentences = inputSentences.map((sentence: string) => {
          const words = sentence.toLowerCase().match(/\b[a-z]{4,}\b/g) || [];
          const unique = [...new Set(words)].slice(0, 3);
          return {
            text: sentence.trim(),
            explanation: `This discusses ${unique.length > 0 ? unique.join(', ') : 'the topic'} and conveys a key idea from the text.`,
          };
        });

        return NextResponse.json({
          explanation: fallbackSentences.map((s: { text: string; explanation: string }) => s.explanation).join(' '),
          sentences: fallbackSentences,
          source: 'extractive',
        });
      }

      case 'chat': {
        if (!chatMessage) {
          return NextResponse.json({ error: 'chatMessage required' }, { status: 400 });
        }

        // MSR-RAG: Multi-Stage Reasoning Retrieval pipeline
        const response = await msrChat(sourceId || null, chatMessage, chatHistory || []);
        return NextResponse.json({ response, source: 'msr-rag' });
      }

      case 'prepare-summary': {
        const cleaned = cleanInputText(text);
        const chunks = chunkForSummarization(cleaned);
        return NextResponse.json({ totalSections: chunks.length, sections: chunks });
      }

      case 'prepare-explain': {
        const sections = chunkForExplain(text);
        return NextResponse.json({ totalSections: sections.length, sections });
      }

      default:
        return NextResponse.json({ error: `Unknown action: ${action}` }, { status: 400 });
    }
  } catch (err) {
    console.error('[API] AI error:', err);
    const msg = err instanceof Error ? err.message : 'AI processing failed';
    const errorText = msg.includes('ECONNREFUSED') || msg.includes('fetch failed')
      ? 'Cannot connect to AI engine. It may still be loading.'
      : msg;
    return NextResponse.json({ error: errorText }, { status: 500 });
  } finally {
    taskCompleted();
  }
}

function parseExplainResponse(
  llmOutput: string,
  originalSentences: string[],
): { text: string; explanation: string }[] {
  const results: { text: string; explanation: string }[] = [];
  const explanationMap = new Map<number, string>();
  const lines = llmOutput.split('\n').filter(l => l.trim());

  for (const line of lines) {
    const match = line.match(/^\[(\d+)\]\s*(.+)/);
    if (match) {
      explanationMap.set(parseInt(match[1], 10), match[2].trim());
    }
  }

  for (let i = 0; i < originalSentences.length; i++) {
    const explanation = explanationMap.get(i + 1);
    if (explanation) {
      results.push({
        text: originalSentences[i].trim(),
        explanation,
      });
    }
  }

  if (results.length === 0 && originalSentences.length > 0) {
    results.push({
      text: originalSentences[0].trim(),
      explanation: llmOutput.replace(/^\[\d+\]\s*/gm, '').trim(),
    });
  }

  return results;
}
