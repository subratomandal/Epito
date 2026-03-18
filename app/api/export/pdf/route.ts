import { NextRequest, NextResponse } from 'next/server';
import { getBrowser, isBrowserAvailable } from '@/inference/browser';

export const dynamic = 'force-dynamic';

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function buildPrintHTML(html: string, title: string): string {
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {
    size: A4;
    margin: 20mm;
  }

  @media print {
    * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
    html, body { margin: 0; padding: 0; }
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #1a1a1a;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  h1 {
    font-size: 22pt;
    font-weight: 700;
    margin: 0 0 14pt 0;
    color: #111;
    line-height: 1.3;
    break-after: avoid;
  }

  h2 {
    font-size: 16pt;
    font-weight: 600;
    margin: 16pt 0 6pt 0;
    color: #111;
    line-height: 1.3;
    break-after: avoid;
  }

  h3 {
    font-size: 13pt;
    font-weight: 600;
    margin: 12pt 0 4pt 0;
    color: #111;
    line-height: 1.4;
    break-after: avoid;
  }

  p { margin: 4pt 0; break-inside: avoid; }
  ul, ol { padding-left: 18pt; margin: 4pt 0; }
  li { margin: 2pt 0; break-inside: avoid; }

  blockquote {
    border-left: 2.5pt solid #ccc;
    padding-left: 10pt;
    color: #555;
    font-style: italic;
    margin: 8pt 0;
    break-inside: avoid;
  }

  code {
    background: #f3f4f6;
    padding: 1pt 4pt;
    border-radius: 2pt;
    font-size: 9.5pt;
    font-family: 'SF Mono', Monaco, Consolas, 'Courier New', monospace;
  }

  pre {
    background: #f3f4f6;
    padding: 10pt;
    border-radius: 4pt;
    overflow-x: auto;
    margin: 8pt 0;
    break-inside: avoid;
  }

  pre code { background: none; padding: 0; font-size: 9.5pt; }

  hr { border: none; border-top: 0.5pt solid #e5e7eb; margin: 14pt 0; }

  a { color: #2563eb; text-decoration: underline; }

  mark { background: #3b82f6; color: #fff; padding: 0 2pt; border-radius: 1pt; }

  img { max-width: 100%; height: auto; border-radius: 4pt; margin: 8pt 0; break-inside: avoid; }

  table { border-collapse: collapse; width: 100%; margin: 8pt 0; break-inside: avoid; }
  td, th { border: 0.5pt solid #e5e7eb; padding: 5pt 8pt; text-align: left; font-size: 10pt; }
  th { background: #f9fafb; font-weight: 600; }

  ul[data-type="taskList"] { list-style: none; padding-left: 0; }
  ul[data-type="taskList"] li { display: flex; align-items: flex-start; gap: 6pt; }
</style>
</head>
<body>
  <h1>${escapeHtml(title || 'Untitled')}</h1>
  ${html}
</body>
</html>`;
}

export async function POST(req: NextRequest) {
  try {
    const { html, title } = await req.json();
    if (!html) return NextResponse.json({ error: 'Missing html' }, { status: 400 });

    if (!isBrowserAvailable()) {
      return NextResponse.json({ error: 'No browser available for PDF export' }, { status: 501 });
    }

    const browser = await getBrowser();
    const page = await browser.newPage();

    try {
      await page.setContent(buildPrintHTML(html, title), { waitUntil: 'domcontentloaded', timeout: 5000 });

      const pdf = await page.pdf({
        format: 'A4',
        margin: { top: '20mm', right: '20mm', bottom: '20mm', left: '20mm' },
        printBackground: true,
        preferCSSPageSize: true,
        scale: 1.0,
      });

      const safeTitle = (title || 'Untitled')
        .replace(/["\r\n\x00-\x1f]/g, '')
        .replace(/[/\\?%*:|<>]/g, '-')
        .slice(0, 100);

      return new NextResponse(Buffer.from(pdf), {
        headers: {
          'Content-Type': 'application/pdf',
          'Content-Disposition': `attachment; filename="${safeTitle}.pdf"`,
        },
      });
    } finally {
      await page.close();
    }
  } catch (err) {
    console.error('[export/pdf] Error:', err);
    const msg = err instanceof Error ? err.message : 'PDF export failed';
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
