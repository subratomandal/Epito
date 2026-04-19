## Epito

An AI-powered, local-first note-taking and document intelligence app with semantic search, OCR, knowledge graph, and an offline desktop assistant.

[![Built with Tauri](https://img.shields.io/badge/Built%20with-Tauri-24C8DB?style=for-the-badge&logo=tauri&logoColor=white)](https://tauri.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](#license)
[![GitHub Stars](https://img.shields.io/github/stars/subratomandal/Epito?style=for-the-badge)](https://github.com/subratomandal/Epito)

### Overview

Epito is a desktop knowledge workspace for writing notes, importing documents and images, extracting text, and asking an on-device AI assistant questions about your content. It combines a TipTap rich text editor, SQLite persistence, MiniLM embeddings, OCR pipelines, and a Tauri-managed llama.cpp runtime so notes, documents, and image text can be searched and reasoned over locally.

Supported Content:

1. Rich text notes
2. PDF documents
3. DOCX documents
4. PNG images
5. JPG / JPEG images
6. Note attachments: PDF, DOCX, TXT, MD, PNG, JPG, JPEG

### Screenshots

<p>
  <img src="https://raw.githubusercontent.com/subratomandal/Epito/main/assets/mainn.png" alt="Epito note editor with AI chat panel" />
</p>

<p>
  <img src="https://raw.githubusercontent.com/subratomandal/Epito/main/assets/ocr.png" alt="Epito image OCR viewer with AI summary panel" />
</p>

<p>
  <img src="https://raw.githubusercontent.com/subratomandal/Epito/main/assets/searchh.png" alt="Epito image OCR viewer with AI summary panel" />
</p>

### Privacy And Security Features

1. Local-first storage: Notes, uploads, embeddings, settings, and caches are stored in SQLite and local files.
2. Local AI inference: The desktop app runs Mistral 7B through `llama-server` on `127.0.0.1` after the model is downloaded.
3. Encrypted settings support: Sensitive settings can be stored with AES-256-GCM using a local 256-bit key.
4. API CSRF checks: Mutating API routes reject unexpected cross-origin requests.
5. Security headers: CSP, `X-Frame-Options`, `nosniff`, referrer policy, and permissions policy are applied by middleware.
6. Upload validation: File type allowlists, request-size limits, generated storage names, and safe file serving are used for uploads.
7. Desktop cleanup: Tauri shuts down Node.js, SQLite, headless browser, and llama-server processes on app exit.
8. On-demand model runtime: llama-server starts only when AI is needed, clears KV cache after inference, and stops after idle time.

### Features

#### Note Workspace

1. Rich text editing with TipTap
2. Autosave with debounced persistence
3. Headings, bold, italic, underline, strikethrough, highlights, links, quotes, rules, task lists, bullet lists, ordered lists, and code blocks
4. Syntax-highlighted code blocks through Lowlight
5. In-note search with highlighted matches
6. Adjustable editor font size
7. Copy selected text or full note text
8. Drag reorder notes in the sidebar
9. Recently deleted notes with restore and permanent delete actions
10. Light and dark themes persisted locally

#### Document And Image Intelligence

1. Upload PDF and DOCX documents
2. Upload PNG, JPG, and JPEG images
3. Extract PDF text layers with `pdf-parse`
4. Extract DOCX text with `mammoth`
5. OCR images with Tesseract fallback and scanned PDFs with optional PaddleOCR
6. Preserve uploaded files in the local data directory
7. View extracted text with virtualized paragraph rendering
8. Search inside extracted document and image text
9. Copy extracted text
10. Zoom image previews in a modal viewer

#### AI Assistance

1. Streamed summaries for notes, documents, and OCR text
2. Section-by-section explanation mode
3. Chat over the selected note, document, or image
4. Hybrid retrieval using embeddings and keyword matching
5. Multi-stage RAG for targeted follow-up retrieval and grounded answers
6. Entity extraction for people, organizations, tools, and relationships
7. Related note discovery through embedding similarity
8. Topic extraction and knowledge graph data
9. Chunk cache for summaries and explanations
10. AI status checks and lazy model startup

#### Search

1. Global search across notes, documents, and images
2. Semantic search with MiniLM embeddings
3. Text fallback search for exact keyword matches
4. Result scoring and matched topic badges
5. Keyboard navigation inside the search dialog

#### Export

1. Export notes as PDF
2. Export notes as DOCX
3. Export notes as PNG image
4. Uses system Chrome or Edge through `puppeteer-core` when available
5. Falls back to `html2canvas` and `jsPDF` when headless browser export is unavailable
6. Uses native save dialogs inside Tauri

#### Keyboard Navigation

1. `Cmd+K` / `Ctrl+K`: Open global search
2. `Cmd+N` / `Ctrl+N`: Create a new note
3. `Cmd+S` / `Ctrl+S`: Save the current note immediately
4. `Cmd+Shift+B` / `Ctrl+Shift+B`: Toggle sidebar collapse
5. `Cmd+F` / `Ctrl+F`: Find inside the current note
6. `Cmd++` / `Ctrl++`: Increase editor font size
7. `Cmd+-` / `Ctrl+-`: Decrease editor font size
8. `Cmd+0` / `Ctrl+0`: Reset editor font size
9. `Escape`: Close active search or export dialogs

### Deployment

#### Desktop App (Recommended)

The packaged app runs as a Tauri desktop application. Tauri starts a bundled Node.js runtime, serves the Next.js standalone app on localhost, and manages the local llama.cpp worker.

```bash
npm install
node scripts/downloadLlamaServer.mjs
npm run tauri:build
```

Build outputs are generated by Tauri under `src-tauri/target/`.

#### macOS

```bash
npm install
node scripts/downloadLlamaServer.mjs

# Apple Silicon
npm run tauri:build:mac:arm

# Universal Intel + Apple Silicon
npm run tauri:build:mac
```

macOS uses the Metal llama.cpp backend and stores packaged app data under `~/.epito/`.

#### Windows

```powershell
npm install
node scripts/downloadLlamaServer.mjs
npm run tauri:build:windows
```

The download script selects CUDA for NVIDIA GPUs, Vulkan for AMD / Intel GPUs, and CPU AVX2 when no supported GPU is detected.

#### Linux

```bash
npm install
node scripts/downloadLlamaServer.mjs
npm run tauri:build:linux
```

Linux builds use the Tauri Linux target and the llama.cpp Ubuntu x64 binary downloaded by the helper script.

#### Local Development

```bash
npm install
npm run dev
```

The Next.js app runs at `http://127.0.0.1:3000`.

For desktop development:

```bash
npm install
node scripts/downloadLlamaServer.mjs
npm run tauri:dev
```

The first desktop launch downloads the Mistral model if it is not already present.

### Environment Variables

1. `EPITO_DATA_DIR` (optional): Data directory for SQLite, uploads, embedding models, OCR cache, and signal files. In web/dev mode the default is `./data`; packaged desktop mode uses `~/.epito/data`.
2. `LLAMA_SERVER_PORT` (optional): Port used by the local llama-server. Default in development is `8080`; packaged desktop builds choose a free port.
3. `PORT` (optional): Next.js server port. Tauri sets this automatically for packaged builds.
4. `HOSTNAME` (optional): Hostname for the Next.js server. Tauri uses `127.0.0.1`.
5. `NODE_ENV` (optional): Runtime environment, usually `development` or `production`.

### AI Model Setup

#### Local LLM

1. Model: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
2. Source: Hugging Face GGUF release by TheBloke
3. Size: about 4.37 GB
4. Storage path: `~/.epito/models`
5. Runtime: `llama-server` from llama.cpp
6. Context size: 4096 tokens

The desktop startup screen downloads the model once and stores it locally.

#### Embeddings

1. Model: `Xenova/all-MiniLM-L6-v2`
2. Dimension: 384
3. Runtime: `@xenova/transformers`
4. Storage path: `EPITO_DATA_DIR/models` or `./data/models`

Embeddings are persisted in SQLite and loaded into an in-memory vector index on startup.

#### OCR

1. PDF text layer: `pdf-parse`
2. DOCX text: `mammoth`
3. Image OCR fallback: `tesseract.js`
4. Scanned PDF OCR: optional PaddleOCR through `scripts/paddleOcr.py`
5. Image preprocessing: `sharp`

PaddleOCR is auto-detected. If it is unavailable, Epito uses Tesseract.js where possible.

### Platform Notes

#### macOS

1. Metal GPU offload for llama.cpp
2. Bundled `.dylib` files under `src-tauri/binaries`
3. Process groups used for clean shutdown
4. User data stored in `~/.epito/`

#### Windows

1. CUDA, Vulkan, or CPU backend selected by GPU detection
2. Job Object cleanup prevents orphan child processes
3. Hidden process creation prevents console flashes
4. WebView2 bootstrapper is embedded by Tauri
5. NSIS installer output is supported

#### Linux

1. llama.cpp Ubuntu x64 binary support
2. System browser detection for high-quality PDF / image export
3. Standard Tauri Linux build flow

### Stack

1. Frontend: Next.js App Router, React, Tailwind CSS
2. Editor: TipTap, ProseMirror, Lowlight
3. Desktop: Tauri v2, Rust
4. Runtime: Bundled Node.js, Next.js standalone output
5. Database: SQLite through `better-sqlite3`
6. AI Runtime: llama.cpp `llama-server`, Mistral 7B Instruct GGUF
7. Embeddings: `@xenova/transformers`, `all-MiniLM-L6-v2`
8. OCR / Parsing: `pdf-parse`, `mammoth`, `tesseract.js`, optional PaddleOCR, `sharp`
9. Export: `puppeteer-core`, `html-to-docx`, `html2canvas`, `jsPDF`
10. Icons: Lucide React

### License

MIT

Created by [@subratomandal](https://github.com/subratomandal)
