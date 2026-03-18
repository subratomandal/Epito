export type ExportFormat = 'pdf' | 'docx' | 'png';

// A4 dimensions
const A4_W = 794;             // 210mm at 96dpi
const A4_H = 1123;            // 297mm at 96dpi
const PAGE_MARGIN = 57;       // ~15mm
const CONTENT_H = A4_H - PAGE_MARGIN * 2;  // 1009px usable height per page
const RENDER_SCALE = 3;       // 3x = ~288 DPI (print-quality, single render keeps it fast)
const PDF_W_MM = 210;
const PDF_H_MM = 297;

// Library preloading — warm the cache when export dialog opens
let _html2canvasPromise: Promise<typeof import('html2canvas')> | null = null;
let _jspdfPromise: Promise<typeof import('jspdf')> | null = null;

function preloadHtml2Canvas() {
  if (!_html2canvasPromise) _html2canvasPromise = import('html2canvas');
  return _html2canvasPromise;
}
function preloadJsPDF() {
  if (!_jspdfPromise) _jspdfPromise = import('jspdf');
  return _jspdfPromise;
}

export function preloadExportLibs(): void {
  preloadHtml2Canvas();
  preloadJsPDF();
}

function sanitizeFilename(name: string): string {
  return (name || 'Untitled')
    .replace(/[/\\?%*:|"<>\r\n\x00-\x1f]/g, '-')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 100);
}

const FORMAT_DESCRIPTIONS: Record<string, string> = {
  pdf: 'PDF Document',
  docx: 'Word Document',
  png: 'PNG Image',
};

function isTauriContext(): boolean {
  try {
    return typeof window !== 'undefined' && '__TAURI__' in window &&
      !!(window as any).__TAURI__?.core?.invoke;
  } catch {
    return false;
  }
}

async function downloadBlob(blob: Blob, filename: string) {
  const ext = filename.split('.').pop() || '';

  if (isTauriContext()) {
    try {
      const { invoke } = (window as any).__TAURI__.core;
      const arrayBuffer = await blob.arrayBuffer();
      const data = Array.from(new Uint8Array(arrayBuffer));

      const saved = await invoke('save_file_with_dialog', {
        data,
        defaultName: filename,
        filterName: FORMAT_DESCRIPTIONS[ext] || 'File',
        filterExtensions: [ext],
      });

      if (saved === true || saved === false) return;
    } catch (err) {
      console.warn('[Export] Tauri save dialog unavailable, using download fallback:', err);
    }
  }

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 200);
}

const EXPORT_CSS = `
  .epito-export { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #1a1a1a; line-height: 1.7; font-size: 15px; }
  .epito-export h1 { font-size: 28px; font-weight: 700; margin: 0 0 20px 0; color: #111; }
  .epito-export h2 { font-size: 21px; font-weight: 600; margin: 18px 0 8px 0; color: #111; }
  .epito-export h3 { font-size: 17px; font-weight: 600; margin: 14px 0 6px 0; color: #111; }
  .epito-export p { margin: 6px 0; }
  .epito-export ul, .epito-export ol { padding-left: 22px; margin: 6px 0; }
  .epito-export li { margin: 3px 0; }
  .epito-export blockquote { border-left: 3px solid #ccc; padding-left: 12px; color: #555; font-style: italic; margin: 10px 0; }
  .epito-export code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 13px; font-family: 'SF Mono', Monaco, Consolas, monospace; }
  .epito-export pre { background: #f3f4f6; padding: 14px; border-radius: 6px; overflow-x: auto; margin: 10px 0; }
  .epito-export pre code { background: none; padding: 0; font-size: 13px; }
  .epito-export hr { border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }
  .epito-export a { color: #2563eb; text-decoration: underline; }
  .epito-export mark { background: #3b82f6; color: #ffffff; padding: 0 2px; border-radius: 2px; }
  .epito-export img { max-width: 100%; height: auto; border-radius: 6px; margin: 10px 0; }
  .epito-export ul[data-type="taskList"] { list-style: none; padding-left: 0; }
  .epito-export ul[data-type="taskList"] li { display: flex; align-items: flex-start; gap: 6px; }
  .epito-export table { border-collapse: collapse; width: 100%; margin: 10px 0; }
  .epito-export td, .epito-export th { border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; }
`;

// --- A4 Page Renderer
// Single html2canvas render of the full document, then slice into A4 pages.
// This is fast because html2canvas traverses the DOM only ONCE, regardless
// of page count. Slicing is cheap canvas-to-canvas copy.

function createExportContainer(html: string, title: string): {
  wrapper: HTMLDivElement;
  inner: HTMLDivElement;
} {
  const wrapper = document.createElement('div');
  wrapper.style.cssText = `position:absolute;left:-9999px;top:0;pointer-events:none;`;

  // A4-width container with margins. Natural height — no clip.
  const container = document.createElement('div');
  container.style.cssText = `width:${A4_W}px;padding:${PAGE_MARGIN}px;box-sizing:border-box;background:white;`;

  // Inner content area (where text flows). Used for break calculation.
  const inner = document.createElement('div');
  inner.classList.add('epito-export');

  const style = document.createElement('style');
  style.textContent = EXPORT_CSS;
  inner.appendChild(style);

  const titleEl = document.createElement('h1');
  titleEl.textContent = title || 'Untitled';
  inner.appendChild(titleEl);

  const body = document.createElement('div');
  body.innerHTML = html;
  inner.appendChild(body);

  container.appendChild(inner);
  wrapper.appendChild(container);

  return { wrapper, inner };
}

// Find natural page break points by snapping to block element boundaries.
function computePageBreaks(inner: HTMLElement): number[] {
  const totalHeight = inner.scrollHeight;
  if (totalHeight <= CONTENT_H) return [0];

  const containerRect = inner.getBoundingClientRect();
  const tops: number[] = [];
  inner.querySelectorAll('p, h1, h2, h3, h4, h5, h6, ul, ol, pre, blockquote, hr, div, table, figure, img, li').forEach(el => {
    const top = Math.round(el.getBoundingClientRect().top - containerRect.top);
    if (top > 0) tops.push(top);
  });
  const uniqueTops = [...new Set(tops)].sort((a, b) => a - b);

  const breaks: number[] = [0];
  let nextIdealBreak = CONTENT_H;

  while (nextIdealBreak < totalHeight) {
    // Find nearest block boundary at or above the ideal cut point
    let bestBreak = nextIdealBreak;
    for (let i = uniqueTops.length - 1; i >= 0; i--) {
      if (uniqueTops[i] <= nextIdealBreak && uniqueTops[i] > nextIdealBreak - CONTENT_H * 0.3) {
        bestBreak = uniqueTops[i];
        break;
      }
    }
    breaks.push(bestBreak);
    nextIdealBreak = bestBreak + CONTENT_H;
  }

  return breaks;
}

// Slice a full-height canvas into A4 pages at the computed break points.
function sliceIntoPages(
  fullCanvas: HTMLCanvasElement,
  breaks: number[],
  scale: number = RENDER_SCALE,
): HTMLCanvasElement[] {
  const scaledW = Math.round(A4_W * scale);
  const scaledPageH = Math.round(A4_H * scale);
  const scaledMargin = Math.round(PAGE_MARGIN * scale);
  const pages: HTMLCanvasElement[] = [];

  for (let i = 0; i < breaks.length; i++) {
    // Break offsets are in inner-content coordinates (no margin).
    // The rendered canvas includes the container's padding, so add margin offset.
    const srcY = Math.round(breaks[i] * RENDER_SCALE + scaledMargin);
    const nextSrcY = i + 1 < breaks.length
      ? Math.round(breaks[i + 1] * RENDER_SCALE + scaledMargin)
      : fullCanvas.height - scaledMargin;
    const sliceH = Math.min(nextSrcY - srcY, scaledPageH - 2 * scaledMargin);

    const page = document.createElement('canvas');
    page.width = scaledW;
    page.height = scaledPageH;
    const ctx = page.getContext('2d');
    if (!ctx) continue;

    // White background (fills the entire A4 page including margins)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, scaledW, scaledPageH);

    // Draw the content slice into the margin area of the page
    ctx.drawImage(
      fullCanvas,
      0, srcY, fullCanvas.width, sliceH,
      scaledMargin, scaledMargin, scaledW - 2 * scaledMargin, sliceH,
    );

    pages.push(page);
  }

  return pages;
}

async function renderA4Pages(html: string, title: string): Promise<HTMLCanvasElement[]> {
  const html2canvas = (await preloadHtml2Canvas()).default;
  const { wrapper, inner } = createExportContainer(html, title);
  document.body.appendChild(wrapper);

  try {
    const breaks = computePageBreaks(inner);
    const container = inner.parentElement!;
    const totalHeight = container.scrollHeight;

    // Auto-reduce scale for very large documents to prevent OOM crashes.
    // ~50M pixels is the practical canvas limit in most browsers.
    const maxPixels = 50_000_000;
    const basePixels = A4_W * totalHeight * RENDER_SCALE * RENDER_SCALE;
    const scale = basePixels > maxPixels
      ? Math.max(1.5, Math.sqrt(maxPixels / (A4_W * totalHeight)))
      : RENDER_SCALE;

    const fullCanvas = await html2canvas(container, {
      scale,
      width: A4_W,
      windowWidth: A4_W,
      backgroundColor: '#ffffff',
      useCORS: true,
      logging: false,
    });

    return sliceIntoPages(fullCanvas, breaks, scale);
  } finally {
    document.body.removeChild(wrapper);
  }
}

// --- PDF Export

export async function exportAsPDF(html: string, title: string): Promise<void> {
  if (!html || !html.trim()) throw new Error('Note is empty. Add content before exporting.');

  const [{ jsPDF }, pages] = await Promise.all([
    preloadJsPDF(),
    renderA4Pages(html, title),
  ]);

  const pdf = new jsPDF('p', 'mm', 'a4');

  for (let i = 0; i < pages.length; i++) {
    if (i > 0) pdf.addPage();
    const imgData = pages[i].toDataURL('image/png');
    pdf.addImage(imgData, 'PNG', 0, 0, PDF_W_MM, PDF_H_MM);
  }

  const blob = pdf.output('blob');
  downloadBlob(blob, sanitizeFilename(title) + '.pdf');
}

// --- Image Export (A4 pages stacked)

export async function exportAsImage(html: string, title: string): Promise<void> {
  if (!html || !html.trim()) throw new Error('Note is empty. Add content before exporting.');

  const pages = await renderA4Pages(html, title);
  if (pages.length === 0) throw new Error('No pages to export.');

  const pageW = pages[0].width;
  const pageH = pages[0].height;
  const gap = 8;
  const totalH = pages.length * pageH + (pages.length - 1) * gap;

  const composite = document.createElement('canvas');
  composite.width = pageW;
  composite.height = totalH;
  const ctx = composite.getContext('2d');
  if (!ctx) throw new Error('Failed to create canvas.');

  ctx.fillStyle = '#e5e7eb';
  ctx.fillRect(0, 0, pageW, totalH);

  for (let i = 0; i < pages.length; i++) {
    ctx.drawImage(pages[i], 0, i * (pageH + gap));
  }

  const blob = await new Promise<Blob>((resolve, reject) => {
    composite.toBlob(b => {
      if (b) resolve(b);
      else reject(new Error('Failed to create image.'));
    }, 'image/png');
  });

  downloadBlob(blob, sanitizeFilename(title) + '.png');
}

// --- DOCX Export

export async function exportAsDOCX(html: string, title: string): Promise<void> {
  if (!html || !html.trim()) throw new Error('Note is empty. Add content before exporting.');

  const res = await fetch('/api/export/docx', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ html, title }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `DOCX export failed (HTTP ${res.status})`);
  }

  const blob = await res.blob();
  downloadBlob(blob, sanitizeFilename(title) + '.docx');
}
