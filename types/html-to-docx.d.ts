declare module 'html-to-docx' {
  interface Options {
    table?: { row?: { cantSplit?: boolean } };
    footer?: boolean;
    pageNumber?: boolean;
    margins?: { top?: number; right?: number; bottom?: number; left?: number };
  }
  export default function HTMLtoDOCX(
    html: string,
    headerHtml: string | null,
    options?: Options,
  ): Promise<Blob>;
}
