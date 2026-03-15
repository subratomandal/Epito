# Windows Platform Notes

## Build
```powershell
# 1. Download GPU-appropriate llama-server (auto-detects NVIDIA/AMD/Intel)
node scripts/downloadLlamaServer.mjs

# 2. Build NSIS installer
npm run tauri:build:windows
```

## Architecture
- GPU: CUDA (NVIDIA) or Vulkan (AMD/Intel) via llama.cpp
- Memory: mmap (OS-managed, demand-paged)
- Node.js: Bundled via externalBin as `node-x86_64-pc-windows-msvc.exe`
- llama-server: Bundled as `llama-server-x86_64-pc-windows-msvc.exe`
- Libraries: `.dll` files in `src-tauri/binaries/`
- CRT: Statically linked (`crt-static` in `.cargo/config.toml`)

## Windows-Specific Behavior
- Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) kills all children on exit
- `CREATE_NO_WINDOW` (0x08000000) on all spawned processes
- `silent_command()` wrapper prevents console flash during GPU detection
- DLL search via PATH env injection (handles Program Files permissions)
- WebView2 via `embedBootstrapper` (works without Edge installed)
- NSIS installer with LZMA compression
- UNC path stripping (`\\?\` prefix removal via `normalize_windows_path`)

## GPU Detection (3-method fallback)
1. `nvidia-smi` — NVIDIA driver tool
2. PowerShell `Get-CimInstance` — modern Windows 10/11
3. WMIC — deprecated fallback

## Download Script GPU Selection
- NVIDIA → CUDA build + CUDA runtime DLLs
- AMD/Intel → Vulkan build
- No GPU → CPU AVX2 build
