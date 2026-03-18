export type ExportFormat = 'pdf' | 'docx' | 'png';

const FORMAT_DESCRIPTIONS: Record<string, string> = {
  pdf: 'PDF Document',
  docx: 'Word Document',
  png: 'PNG Image',
};

function sanitizeFilename(name: string): string {
  return (name || 'Untitled')
    .replace(/[/\\?%*:|"<>\r\n\x00-\x1f]/g, '-')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 100);
}

function isTauriContext(): boolean {
  try {
    return typeof window !== 'undefined' && '__TAURI__' in window &&
      !!(window as any).__TAURI__?.core?.invoke;
  } catch { return false; }
}

async function downloadBlob(blob: Blob, filename: string): Promise<void> {
  const ext = filename.split('.').pop() || '';

  if (isTauriContext()) {
    try {
      const { invoke } = (window as any).__TAURI__.core;
      const bytes = new Uint8Array(await blob.arrayBuffer());
      // Pass raw bytes array — Tauri deserializes Vec<u8> from this
      const saved = await invoke('save_file_with_dialog', {
        data: Array.from(bytes),
        defaultName: filename,
        filterName: FORMAT_DESCRIPTIONS[ext] || 'File',
        filterExtensions: [ext],
      });
      if (saved === true || saved === false) return;
    } catch (err) {
      console.warn('[Export] Tauri save dialog error, using download fallback:', err);
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

export function preloadExportLibs(): void {
  // Warm html2canvas cache
  import('html2canvas').catch(() => {});
}

// --- Headless Chrome check (one-time, cached)
let chromeChecked = false;
let chromeWorks = false;

async function tryChromePDF(html: string, title: string): Promise<Blob | null> {
  if (chromeChecked && !chromeWorks) return null;

  try {
    const res = await fetch('/api/export/pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ html, title }),
      signal: AbortSignal.timeout(8000),
    });
    chromeChecked = true;
    if (res.ok) {
      chromeWorks = true;
      return await res.blob();
    }
    chromeWorks = false;
    return null;
  } catch {
    chromeChecked = true;
    chromeWorks = false;
    return null;
  }
}

async function tryChromeImage(html: string, title: string): Promise<Blob | null> {
  if (chromeChecked && !chromeWorks) return null;

  try {
    const res = await fetch('/api/export/image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ html, title }),
      signal: AbortSignal.timeout(8000),
    });
    chromeChecked = true;
    if (res.ok) {
      chromeWorks = true;
      return await res.blob();
    }
    chromeWorks = false;
    return null;
  } catch {
    chromeChecked = true;
    chromeWorks = false;
    return null;
  }
}

// --- PDF Export

export async function exportAsPDF(html: string, title: string): Promise<void> {
  if (!html || !html.trim()) throw new Error('Note is empty. Add content before exporting.');

  // Try headless Chrome (vector PDF, best quality, skipped if previously failed)
  const chromeBlob = await tryChromePDF(html, title);
  if (chromeBlob) {
    await downloadBlob(chromeBlob, sanitizeFilename(title) + '.pdf');
    return;
  }

  // Canvas fallback
  const pages = await renderPages(html, title);
  const jspdfModule = await import('jspdf');
  const pdf = new jspdfModule.jsPDF('p', 'mm', 'a4');

  for (let i = 0; i < pages.length; i++) {
    if (i > 0) pdf.addPage();
    pdf.addImage(pages[i].toDataURL('image/png'), 'PNG', 0, 0, 210, 297);
  }

  await downloadBlob(pdf.output('blob'), sanitizeFilename(title) + '.pdf');
}

// --- Image Export

export async function exportAsImage(html: string, title: string): Promise<void> {
  if (!html || !html.trim()) throw new Error('Note is empty. Add content before exporting.');

  const chromeBlob = await tryChromeImage(html, title);
  if (chromeBlob) {
    await downloadBlob(chromeBlob, sanitizeFilename(title) + '.png');
    return;
  }

  const pages = await renderPages(html, title);
  if (pages.length === 0) throw new Error('No pages.');

  const pw = pages[0].width, ph = pages[0].height, gap = 8;
  const c = document.createElement('canvas');
  c.width = pw;
  c.height = pages.length * ph + (pages.length - 1) * gap;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#e5e7eb';
  ctx.fillRect(0, 0, c.width, c.height);
  for (let i = 0; i < pages.length; i++) ctx.drawImage(pages[i], 0, i * (ph + gap));

  const blob = await new Promise<Blob>((resolve, reject) => {
    c.toBlob(b => b ? resolve(b) : reject(new Error('Image failed')), 'image/png');
  });
  await downloadBlob(blob, sanitizeFilename(title) + '.png');
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
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `DOCX export failed (HTTP ${res.status})`);
  }

  await downloadBlob(await res.blob(), sanitizeFilename(title) + '.docx');
}

// --- Canvas rendering (fast: single render, scale 3, page-break slicing)

const A4_W = 794;
const A4_H = 1123;
const MARGIN = 57;
const CONTENT_H = A4_H - MARGIN * 2;
const SCALE = 3;

const CSS = `
  .ee { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',sans-serif; color:#1a1a1a; line-height:1.7; font-size:15px; -webkit-font-smoothing:antialiased; }
  .ee h1 { font-size:26px; font-weight:700; margin:0 0 16px; color:#111; line-height:1.3; }
  .ee h2 { font-size:21px; font-weight:600; margin:14px 0 8px; color:#111; }
  .ee h3 { font-size:17px; font-weight:600; margin:12px 0 6px; color:#111; }
  .ee p { margin:6px 0; } .ee ul,.ee ol { padding-left:22px; margin:6px 0; } .ee li { margin:3px 0; }
  .ee blockquote { border-left:3px solid #ccc; padding-left:12px; color:#555; font-style:italic; margin:10px 0; }
  .ee code { background:#f3f4f6; padding:1px 5px; border-radius:3px; font-size:13px; font-family:'SF Mono',Monaco,Consolas,monospace; }
  .ee pre { background:#f3f4f6; padding:14px; border-radius:6px; overflow-x:auto; margin:10px 0; }
  .ee pre code { background:none; padding:0; }
  .ee hr { border:none; border-top:1px solid #e5e7eb; margin:20px 0; }
  .ee a { color:#2563eb; text-decoration:underline; }
  .ee mark { background:#3b82f6; color:#fff; padding:0 2px; border-radius:2px; }
  .ee img { max-width:100%; height:auto; border-radius:6px; margin:10px 0; }
  .ee table { border-collapse:collapse; width:100%; margin:10px 0; }
  .ee td,.ee th { border:1px solid #e5e7eb; padding:6px 10px; text-align:left; }
`;

async function renderPages(html: string, title: string): Promise<HTMLCanvasElement[]> {
  const html2canvas = (await import('html2canvas')).default;

  const w = document.createElement('div');
  w.style.cssText = 'position:absolute;left:-9999px;top:0;pointer-events:none;';
  const box = document.createElement('div');
  box.style.cssText = `width:${A4_W}px;padding:${MARGIN}px;box-sizing:border-box;background:white;`;
  const inner = document.createElement('div');
  inner.classList.add('ee');
  inner.innerHTML = `<style>${CSS}</style><h1>${(title || 'Untitled').replace(/</g, '&lt;')}</h1>${html}`;
  box.appendChild(inner);
  w.appendChild(box);
  document.body.appendChild(w);

  try {
    // Page breaks
    const breaks: number[] = [0];
    const th = inner.scrollHeight;
    if (th > CONTENT_H) {
      const rect = inner.getBoundingClientRect();
      const tops = new Set<number>();
      inner.querySelectorAll('p,h1,h2,h3,h4,h5,h6,ul,ol,pre,blockquote,hr,div,table,img,li').forEach(el => {
        const t = Math.round(el.getBoundingClientRect().top - rect.top);
        if (t > 0) tops.add(t);
      });
      const sorted = [...tops].sort((a, b) => a - b);
      let next = CONTENT_H;
      while (next < th) {
        let best = next;
        for (let i = sorted.length - 1; i >= 0; i--) {
          if (sorted[i] <= next && sorted[i] > next - CONTENT_H * 0.3) { best = sorted[i]; break; }
        }
        breaks.push(best);
        next = best + CONTENT_H;
      }
    }

    // Single render
    const full = await html2canvas(box, {
      scale: SCALE, width: A4_W, windowWidth: A4_W,
      backgroundColor: '#ffffff', useCORS: true, logging: false,
    });

    // Slice
    const sw = Math.round(A4_W * SCALE);
    const sh = Math.round(A4_H * SCALE);
    const sm = Math.round(MARGIN * SCALE);
    const pages: HTMLCanvasElement[] = [];

    for (let i = 0; i < breaks.length; i++) {
      const y0 = Math.round(breaks[i] * SCALE + sm);
      const y1 = i + 1 < breaks.length ? Math.round(breaks[i + 1] * SCALE + sm) : full.height - sm;
      const sl = Math.min(y1 - y0, sh - 2 * sm);

      const pg = document.createElement('canvas');
      pg.width = sw; pg.height = sh;
      const ctx = pg.getContext('2d')!;
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, sw, sh);
      ctx.drawImage(full, 0, y0, full.width, sl, sm, sm, sw - 2 * sm, sl);
      pages.push(pg);
    }

    return pages;
  } finally {
    document.body.removeChild(w);
  }
}
