import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const response = NextResponse.next();

  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');

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
