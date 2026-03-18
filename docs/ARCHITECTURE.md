# Epito Architecture

## Directory Structure

```
epito/
├── app/                     Next.js App Router (pages + API routes)
│   └── api/                 REST API endpoints
│
├── components/              React UI
│   ├── editor/              Rich text editor (TipTap)
│   ├── sidebar/             Note/doc/image list
│   ├── viewer/              Document & image viewer
│   ├── search/              Global search dialog
│   ├── related/             AI panel (summary, explain, chat)
│   ├── common/              Splash, error boundary
│   ├── startup/             Model download screen
│   └── ui/                  Shared primitives
│
├── model/                   LLM inference runtime
│   ├── llm.ts               Llama-server management, idle offload, chat pipeline
│   ├── explain.ts            Sequential explanation pipeline
│   └── answer-engine.ts      Context extraction for RAG
│
├── inference/               Request handling + generation pipeline
│   ├── pipeline.ts           Semantic search, chunking, embeddings, topic extraction
│   ├── lifecycle.ts          Shutdown coordination, task concurrency
│   ├── ocr.ts                Text extraction (PaddleOCR, Tesseract, pdf-parse)
│   └── export.ts             PDF / DOCX / PNG export (A4 pagination)
│
├── memory/                  Embeddings + retrieval
│   ├── embeddings.ts         all-MiniLM-L6-v2 via @xenova/transformers
│   └── vector.ts             In-memory vector index, cosine similarity
│
├── notes/                   Note storage + indexing
│   ├── database.ts           SQLite (better-sqlite3), full CRUD
│   └── encryption.ts         AES-256-GCM for sensitive settings
│
├── common/                  Shared types + utilities
│   ├── types.ts              TypeScript interfaces (Note, Document, etc.)
│   ├── utils.ts              cn(), debounce(), stripHtml()
│   └── html-to-docx.d.ts     External type declaration
│
├── src-tauri/               Tauri desktop shell (Rust)
│   └── src/
│       ├── lib.rs            App entry, Node.js spawning, shutdown
│       ├── llama_server.rs   LLM process management, GPU detection
│       ├── model.rs          Model download/management
│       └── native_win.rs     Windows APIs (DWM, DPI, VRAM, power)
│
├── scripts/                 Build & dev automation
├── docs/                    Architecture, platform docs
├── public/                  Static assets
└── middleware.ts            CSRF protection, security headers
```

## Data Flow

```
User → TipTap Editor → Debounced Save → PUT /api/notes/:id → SQLite
                                              ↓
                                        inference/pipeline (background)
                                              ↓
                                    Chunking → memory/embeddings → Vector Index
```

## AI Architecture

- **On-demand**: LLM starts only when AI task is triggered, not at boot
- **Instant offload**: KV cache cleared immediately after inference, process killed after 30s idle
- **Signal-based**: Node.js writes signal files, Rust polls at 200ms
- **GPU-aware**: CUDA > Vulkan > CPU, layer count calculated from VRAM
