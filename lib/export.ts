export type ExportFormat = 'pdf' | 'docx' | 'png';

function buildStyledHTML(html: string, title: string): string {
  return `
    <div style="
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      color: #1a1a1a;
      line-height: 1.8;
      max-width: 1000px;
      margin: 0 auto;
      padding: 60px;
      font-size: 16px;
    ">
      <h1 style="font-size: 32px; font-weight: 700; margin-bottom: 28px; color: #111;">${escapeHtml(title)}</h1>
      ${html}
    </div>
  `;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

const MIME_TYPES: Record<string, string> = {
  pdf: 'application/pdf',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  png: 'image/png',
};

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
      const bytes = new Uint8Array(arrayBuffer);
      const data: number[] = new Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) data[i] = bytes[i];

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
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 200);
}

async function renderToCanvas(html: string, title: string): Promise<HTMLCanvasElement> {
  const html2canvas = (await import('html2canvas')).default;

  const wrapper = document.createElement('div');
  wrapper.style.cssText = `position:absolute;left:-9999px;top:0;width:0;height:0;overflow:hidden;pointer-events:none;`;

  const container = document.createElement('div');
  container.style.cssText = `width:1200px;background:white;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;`;
  container.innerHTML = buildStyledHTML(html, title);

  const styles = document.createElement('style');
  styles.textContent = `
    .epito-export h1, .epito-export h2, .epito-export h3 { color: #111; margin-top: 1em; margin-bottom: 0.5em; }
    .epito-export h2 { font-size: 22px; } .epito-export h3 { font-size: 18px; }
    .epito-export p { margin: 8px 0; font-size: 16px; line-height: 1.8; }
    .epito-export ul, .epito-export ol { padding-left: 24px; margin: 8px 0; }
    .epito-export li { margin: 4px 0; font-size: 16px; }
    .epito-export blockquote { border-left: 3px solid #ccc; padding-left: 12px; color: #555; font-style: italic; margin: 12px 0; }
    .epito-export code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-size: 14px; font-family: 'SF Mono', Monaco, Consolas, monospace; }
    .epito-export pre { background: #f3f4f6; padding: 16px; border-radius: 8px; overflow-x: auto; margin: 12px 0; }
    .epito-export pre code { background: none; padding: 0; font-size: 14px; }
    .epito-export hr { border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; }
    .epito-export a { color: #2563eb; text-decoration: underline; }
    .epito-export mark { background: rgba(250, 204, 21, 0.4); padding: 0 2px; border-radius: 2px; }
    .epito-export img { max-width: 100%; height: auto; border-radius: 8px; margin: 12px 0; }
    .epito-export ul[data-type="taskList"] { list-style: none; padding-left: 0; }
    .epito-export ul[data-type="taskList"] li { display: flex; align-items: flex-start; gap: 8px; }
  `;
  container.classList.add('epito-export');
  container.prepend(styles);
  wrapper.appendChild(container);
  document.body.appendChild(wrapper);

  try {
    const canvas = await html2canvas(container, {
      scale: 3,
      useCORS: true,
      backgroundColor: '#ffffff',
      width: 1200,
      windowWidth: 1200,
    });
    return canvas;
  } finally {
    document.body.removeChild(wrapper);
  }
}

export async function exportAsPDF(html: string, title: string): Promise<void> {
  if (!html || !html.trim()) throw new Error('Note is empty. Add content before exporting.');

  const { jsPDF } = await import('jspdf');
  const canvas = await renderToCanvas(html, title);

  const imgWidth = 210;
  const pageHeight = 297;
  const margin = 10;
  const contentWidth = imgWidth - margin * 2;
  const imgHeight = (canvas.height * contentWidth) / canvas.width;

  const pdf = new jsPDF('p', 'mm', 'a4');
  const pageContentHeight = pageHeight - margin * 2;

  if (imgHeight <= pageContentHeight) {
    pdf.addImage(canvas.toDataURL('image/png'), 'PNG', margin, margin, contentWidth, imgHeight);
  } else {
    let yOffset = 0;
    let page = 0;
    const scaleFactor = canvas.width / contentWidth;

    while (yOffset < canvas.height) {
      if (page > 0) pdf.addPage();

      const sliceHeight = Math.min(pageContentHeight * scaleFactor, canvas.height - yOffset);
      const sliceCanvas = document.createElement('canvas');
      sliceCanvas.width = canvas.width;
      sliceCanvas.height = sliceHeight;
      const ctx = sliceCanvas.getContext('2d');
      if (!ctx) throw new Error('Failed to create canvas context for PDF page.');
      ctx.drawImage(canvas, 0, yOffset, canvas.width, sliceHeight, 0, 0, canvas.width, sliceHeight);

      const sliceImgHeight = (sliceHeight * contentWidth) / canvas.width;
      pdf.addImage(sliceCanvas.toDataURL('image/png'), 'PNG', margin, margin, contentWidth, sliceImgHeight);

      yOffset += sliceHeight;
      page++;
    }
  }

  const blob = pdf.output('blob');
  downloadBlob(blob, sanitizeFilename(title) + '.pdf');
}

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

export async function exportAsImage(html: string, title: string): Promise<void> {
  if (!html || !html.trim()) throw new Error('Note is empty. Add content before exporting.');

  const canvas = await renderToCanvas(html, title);

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((b) => {
      if (b) resolve(b);
      else reject(new Error('Failed to create image. The note content may be too large.'));
    }, 'image/png');
  });

  downloadBlob(blob, sanitizeFilename(title) + '.png');
}

function sanitizeFilename(name: string): string {
  return (name || 'Untitled')
    .replace(/[/\\?%*:|"<>]/g, '-')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 100);
}
