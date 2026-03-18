import { NextRequest, NextResponse } from 'next/server';

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export async function POST(req: NextRequest) {
  try {
    const { html, title } = await req.json();
    if (!html) {
      return NextResponse.json({ error: 'Missing html' }, { status: 400 });
    }

    const HTMLtoDOCX = (await import('html-to-docx')).default;

    const fullHtml = `<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body>
  <h1>${escapeHtml(title || 'Untitled')}</h1>
  ${html}
</body></html>`;

    const buffer = await HTMLtoDOCX(fullHtml, null, {
      table: { row: { cantSplit: true } },
      footer: true,
      pageNumber: true,
      page: {
        size: { width: '210mm', height: '297mm' },
        margin: { top: '15mm', right: '15mm', bottom: '15mm', left: '15mm' },
      },
      font: 'Calibri',
      fontSize: 22, // half-points: 22 = 11pt
    });

    const arrayBuffer = buffer instanceof Buffer
      ? buffer
      : Buffer.from(await (buffer as Blob).arrayBuffer());

    const safeTitle = (title || 'Untitled')
      .replace(/["\r\n\x00-\x1f]/g, '')
      .replace(/[/\\?%*:|<>]/g, '-')
      .slice(0, 100);

    return new NextResponse(arrayBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'Content-Disposition': `attachment; filename="${safeTitle}.docx"`,
      },
    });
  } catch (err) {
    console.error('[export/docx] Error:', err);
    return NextResponse.json({ error: 'Export failed' }, { status: 500 });
  }
}
