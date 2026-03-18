import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // CSRF protection: block cross-origin mutating requests to API endpoints
  if (request.nextUrl.pathname.startsWith('/api/')) {
    const method = request.method.toUpperCase();
    if (method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS') {
      const origin = request.headers.get('origin') || '';
      const referer = request.headers.get('referer') || '';

      // Allow: no origin (server-side/curl), localhost, tauri://
      const isAllowedOrigin =
        !origin ||
        origin.startsWith('http://127.0.0.1') ||
        origin.startsWith('http://localhost') ||
        origin.startsWith('tauri://');
      const isAllowedReferer =
        !referer ||
        referer.startsWith('http://127.0.0.1') ||
        referer.startsWith('http://localhost') ||
        referer.startsWith('tauri://');

      if (!isAllowedOrigin && !isAllowedReferer) {
        return NextResponse.json({ error: 'Forbidden: cross-origin request' }, { status: 403 });
      }
    }
  }

  const response = NextResponse.next();

  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  response.headers.set(
    'Content-Security-Policy',
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self' http://127.0.0.1:* http://localhost:*; img-src 'self' data: blob: http://127.0.0.1:* http://localhost:*; font-src 'self' data:; object-src 'none'; base-uri 'self'; worker-src 'self'; frame-ancestors 'none';"
  );

  const contentLength = request.headers.get('content-length');
  if (contentLength && request.nextUrl.pathname.startsWith('/api/')) {
    const size = parseInt(contentLength, 10);
    const isUpload = request.nextUrl.pathname.startsWith('/api/upload') || request.nextUrl.pathname.startsWith('/api/attachments');
    const maxSize = isUpload ? 50 * 1024 * 1024 : 1 * 1024 * 1024;
    if (size > maxSize) {
      return NextResponse.json({ error: 'Request too large' }, { status: 413 });
    }
  }

  return response;
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
