import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Epito',
  description: 'AI-powered note-taking with semantic search and knowledge graph',
  manifest: '/manifest.json',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Epito',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#0a0a0f',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" type="image/x-icon" href="/favicon.ico" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon.png" />
        <link rel="apple-touch-icon" href="/logo.png" />
        <meta name="mobile-web-app-capable" content="yes" />
        <script dangerouslySetInnerHTML={{
          __html: `(function(){var p=new URLSearchParams(window.location.search).get('theme');var t=p||localStorage.getItem('theme')||'dark';if(p)localStorage.setItem('theme',p);document.documentElement.classList.add(t)})()`,
        }} />
        <script dangerouslySetInnerHTML={{
          __html: `if('serviceWorker' in navigator){window.addEventListener('load',function(){navigator.serviceWorker.register('/sw.js')})}`,
        }} />
        <script dangerouslySetInnerHTML={{
          __html: `window.addEventListener('beforeunload',function(){document.documentElement.classList.add('app-closing')});if(window.__TAURI_INTERNALS__){window.__TAURI_INTERNALS__.invoke('plugin:event|listen',{event:'tauri://close-requested',handler:0}).catch(function(){})}`,
        }} />
      </head>
      <body className={`${inter.className} bg-background text-foreground antialiased`}>
        {children}
      </body>
    </html>
  );
}
