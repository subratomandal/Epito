import { NextRequest, NextResponse } from 'next/server';
import { getBrowser, isBrowserAvailable } from '@/inference/browser';

export const dynamic = 'force-dynamic';

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function buildImageHTML(html: string, title: string): string {
  // A4 at base width, rendered at 2x deviceScaleFactor for retina output
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    width: 794px;
    padding: 57px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    font-size: 15px;
    line-height: 1.7;
    color: #1a1a1a;
    background: #ffffff;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  h1 { font-size: 26px; font-weight: 700; margin: 0 0 16px 0; color: #111; line-height: 1.3; }
  h2 { font-size: 21px; font-weight: 600; margin: 14px 0 8px 0; color: #111; line-height: 1.3; }
  h3 { font-size: 17px; font-weight: 600; margin: 12px 0 6px 0; color: #111; line-height: 1.4; }
  p { margin: 6px 0; }
  ul, ol { padding-left: 22px; margin: 6px 0; }
  li { margin: 3px 0; }
  blockquote { border-left: 3px solid #ccc; padding-left: 12px; color: #555; font-style: italic; margin: 10px 0; }
  code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 13px; font-family: 'SF Mono', Monaco, Consolas, monospace; }
  pre { background: #f3f4f6; padding: 14px; border-radius: 6px; overflow-x: auto; margin: 10px 0; }
  pre code { background: none; padding: 0; font-size: 13px; }
  hr { border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }
  a { color: #2563eb; text-decoration: underline; }
  mark { background: #3b82f6; color: #fff; padding: 0 2px; border-radius: 2px; }
  img { max-width: 100%; height: auto; border-radius: 6px; margin: 10px 0; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; }
  td, th { border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; }
  th { background: #f9fafb; font-weight: 600; }
  ul[data-type="taskList"] { list-style: none; padding-left: 0; }
  ul[data-type="taskList"] li { display: flex; align-items: flex-start; gap: 6px; }
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
      return NextResponse.json({ error: 'No browser available for image export' }, { status: 501 });
    }

    const browser = await getBrowser();
    const page = await browser.newPage();

    try {
      await page.setViewport({ width: 794, height: 1123, deviceScaleFactor: 3 });
      await page.setContent(buildImageHTML(html, title), { waitUntil: 'domcontentloaded', timeout: 5000 });

      const screenshot = await page.screenshot({
        type: 'png',
        fullPage: true,
        omitBackground: false,
      });

      const safeTitle = (title || 'Untitled')
        .replace(/["\r\n\x00-\x1f]/g, '')
        .replace(/[/\\?%*:|<>]/g, '-')
        .slice(0, 100);

      return new NextResponse(Buffer.from(screenshot), {
        headers: {
          'Content-Type': 'image/png',
          'Content-Disposition': `attachment; filename="${safeTitle}.png"`,
        },
      });
    } finally {
      await page.close();
    }
  } catch (err) {
    console.error('[export/image] Error:', err);
    const msg = err instanceof Error ? err.message : 'Image export failed';
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
