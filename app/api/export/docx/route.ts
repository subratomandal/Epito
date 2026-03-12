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
    });

    const arrayBuffer = buffer instanceof Buffer
      ? buffer
      : Buffer.from(await (buffer as Blob).arrayBuffer());

    return new NextResponse(arrayBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'Content-Disposition': `attachment; filename="${(title || 'Untitled').replace(/"/g, '')}.docx"`,
      },
    });
  } catch (err) {
    console.error('[export/docx] Error:', err);
    return NextResponse.json({ error: 'Export failed' }, { status: 500 });
  }
}
