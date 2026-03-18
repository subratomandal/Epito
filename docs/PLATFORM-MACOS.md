# macOS Platform Notes

## Build
```bash
npm run tauri:build:mac:arm    # Apple Silicon
npm run tauri:build:mac        # Universal (Intel + Apple Silicon)
```

## Architecture
- GPU: Metal backend via llama.cpp (compiled into the macOS binary)
- Memory: mmap (OS-managed, demand-paged)
- Node.js: Bundled via externalBin as `node-aarch64-apple-darwin`
- llama-server: Bundled as `llama-server-aarch64-apple-darwin`
- Libraries: `.dylib` files in `src-tauri/binaries/`

## macOS-Specific Behavior
- Metal GPU offload with `--n-gpu-layers 99 --flash-attn`
- Process groups via `process_group(0)` for clean SIGTERM/SIGKILL
- Unix permissions auto-fixed on bundled binaries
- `.epito/` data directory in user home
