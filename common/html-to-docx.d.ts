declare module 'html-to-docx' {
  interface Options {
    table?: { row?: { cantSplit?: boolean } };
    footer?: boolean;
    pageNumber?: boolean;
    margins?: { top?: number; right?: number; bottom?: number; left?: number };
    page?: {
      size?: { width?: string | number; height?: string | number };
      margin?: { top?: string | number; right?: string | number; bottom?: string | number; left?: string | number };
    };
    font?: string;
    fontSize?: number;
  }
  export default function HTMLtoDOCX(
    html: string,
    headerHtml: string | null,
    options?: Options,
  ): Promise<Blob>;
}
