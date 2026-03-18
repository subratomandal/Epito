export type ExportFormat = 'pdf' | 'docx' | 'png';

// A4 page dimensions (96 DPI)
const A4_W = 794;            // 210mm at 96dpi
const A4_H = 1123;           // 297mm at 96dpi
const PAGE_MARGIN = 57;      // ~15mm
const CONTENT_W = A4_W - PAGE_MARGIN * 2;   // 680px
const CONTENT_H = A4_H - PAGE_MARGIN * 2;   // 1009px
const RENDER_SCALE = 2;      // 2x = ~190 DPI (crisp, fast)

// A4 PDF dimensions in mm
const PDF_W_MM = 210;
const PDF_H_MM = 297;

// Library preloading — warm the cache before the user clicks export
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

/** Call when the export dialog opens. */
export function preloadExportLibs(): void {
  preloadHtml2Canvas();
  preloadJsPDF();
}

// Helpers

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

// Export stylesheet

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

// Per-page A4 renderer — renders one page at a time to avoid massive canvases,
// snapping breaks to block boundaries so text is never sliced mid-paragraph.

function createExportDOM(html: string, title: string): {
  clip: HTMLDivElement;
  content: HTMLDivElement;
  cleanup: () => void;
} {
  const clip = document.createElement('div');
  clip.style.cssText = `position:absolute;left:-9999px;top:0;width:${A4_W}px;height:${A4_H}px;overflow:hidden;background:white;pointer-events:none;`;

  const frame = document.createElement('div');
  frame.style.cssText = `padding:${PAGE_MARGIN}px;width:${A4_W}px;box-sizing:border-box;`;

  const content = document.createElement('div');
  content.style.cssText = `width:${CONTENT_W}px;background:white;`;
  content.classList.add('epito-export');

  const style = document.createElement('style');
  style.textContent = EXPORT_CSS;
  content.appendChild(style);

  const titleEl = document.createElement('h1');
  titleEl.textContent = title || 'Untitled';
  content.appendChild(titleEl);

  const body = document.createElement('div');
  body.innerHTML = html;
  content.appendChild(body);

  frame.appendChild(content);
  clip.appendChild(frame);
  document.body.appendChild(clip);

  return {
    clip,
    content,
    cleanup: () => document.body.removeChild(clip),
  };
}

/** Snap page breaks to block element boundaries to avoid cutting mid-paragraph. */
function computePageBreaks(content: HTMLElement): number[] {
  const totalHeight = content.scrollHeight;
  if (totalHeight <= CONTENT_H) return [0];

  const containerRect = content.getBoundingClientRect();
  const tops: number[] = [];
  content.querySelectorAll('p, h1, h2, h3, h4, h5, h6, ul, ol, pre, blockquote, hr, div, table, figure, img, li').forEach(el => {
    const top = Math.round(el.getBoundingClientRect().top - containerRect.top);
    if (top > 0) tops.push(top);
  });
  const uniqueTops = [...new Set(tops)].sort((a, b) => a - b);

  const breaks: number[] = [0];
  let nextIdealBreak = CONTENT_H;

  while (nextIdealBreak < totalHeight) {
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

async function renderA4Pages(html: string, title: string): Promise<HTMLCanvasElement[]> {
  const html2canvas = (await preloadHtml2Canvas()).default;
  const { clip, content, cleanup } = createExportDOM(html, title);

  try {
    const breaks = computePageBreaks(content);
    const pages: HTMLCanvasElement[] = [];

    for (let i = 0; i < breaks.length; i++) {
      const offset = breaks[i];
      content.style.marginTop = `${-offset}px`;
      content.offsetHeight; // eslint-disable-line @typescript-eslint/no-unused-expressions -- force layout before capture

      const canvas = await html2canvas(clip, {
        scale: RENDER_SCALE,
        width: A4_W,
        height: A4_H,
        windowWidth: A4_W,
        backgroundColor: '#ffffff',
        useCORS: true,
        logging: false,
      });
      pages.push(canvas);
    }

    return pages;
  } finally {
    cleanup();
  }
}

// PDF export

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

// Image export (A4 pages stacked)

export async function exportAsImage(html: string, title: string): Promise<void> {
  if (!html || !html.trim()) throw new Error('Note is empty. Add content before exporting.');

  const pages = await renderA4Pages(html, title);

  const pageW = A4_W * RENDER_SCALE;
  const pageH = A4_H * RENDER_SCALE;
  const gap = 4 * RENDER_SCALE;
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

// DOCX export

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
