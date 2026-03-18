import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type DebouncedFn<T extends (...args: any[]) => any> = // eslint-disable-line @typescript-eslint/no-explicit-any
  ((...args: Parameters<T>) => void) & { cancel: () => void; flush: (...args: Parameters<T>) => void };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): DebouncedFn<T> {
  let timer: ReturnType<typeof setTimeout>;
  let lastArgs: Parameters<T> | null = null;
  const debounced = ((...args: Parameters<T>) => {
    clearTimeout(timer);
    lastArgs = args;
    timer = setTimeout(() => { lastArgs = null; fn(...args); }, delay);
  }) as DebouncedFn<T>;
  debounced.cancel = () => { clearTimeout(timer); lastArgs = null; };
  debounced.flush = (...args: Parameters<T>) => {
    clearTimeout(timer);
    const a = args.length ? args : lastArgs;
    lastArgs = null;
    if (a) fn(...a);
  };
  return debounced;
}

export function stripHtml(html: string): string {
  return html
    .replace(/<[^>]*>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/\s+/g, ' ')
    .trim();
}

export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).trimEnd() + '...';
}
