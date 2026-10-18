# AI Context: RSNA Knee Abnormality Detection

## 1. Project Scope
*   **Competition:** RSNA Knee Abnormality Detection
*   **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview
*   **Objective:** Multi-modal 3D medical imaging and NLP pipeline to predict 12 per-study abnormality labels.
*   **Plan and TODO:** See TODO.ORG in root of repo

## 2. Hardware & Resource Constraints
*   **GPU:** 1x NVIDIA RTX 5080 (16GB VRAM). VRAM is the primary bottleneck. Prioritize gradient checkpointing, mixed precision (FP8/BF16), and memory-efficient batching.
*   **CPU:** AMD Ryzen 7 9850X3D (8 Cores / 16 Threads). Avoid CPU starvation. Limit `DataLoader` workers and push heavy processing to an offline caching phase.
*   **Storage:** High-speed Gen4 NVMe. Optimize for sequential reads by utilizing chunked binary storage (e.g., Zarr, HDF5, or memory-mapped `.npy`) rather than iterating over thousands of raw `.dcm` files.

## 3. Operating System & Workflow
*   **OS:** Linux (CachyOS / Void Linux). Assume a terminal-first workflow. Shell commands should be POSIX-compliant or standard Linux utilities. 
*   **Editors (Dual-Wield):** **Doom Emacs** (Evil mode) and **Google Antigravity IDE**. Provide clean, raw code blocks easily yanked into Emacs buffers, while remaining compatible with Antigravity's agentic workflows for parallel execution and visual verification.
*   **Backups:** Remind me to initiate BorgBackup snapshots via systemd timers before executing destructive data pipeline migrations or clearing large tensor caches.

## 4. Coding Standards & Best Practices
*   **Style:** Strictly adhere to PEP 8 guidelines. Use standard Python tooling (e.g., `black`, `ruff`, `isort`) for formatting.
*   **Data Handling:** Leverage `pandas` for tabular data, metadata processing, and file path lookups. Ensure database connections (if utilizing SQLAlchemy or local SQLite for logging) are safely closed and pooled.
*   **Modularity:** Keep scripts modular. Decouple data parsing, model definition, and training loops. 
*   **Error Handling:** Implement defensive programming. Include explicit exception handling for corrupted DICOM files or OOM (Out of Memory) CUDA errors.

## 5. Output Preferences
*   Provide complete, runnable scripts rather than disjointed snippets when implementing complex logic.
*   If a hardware limitation forces a compromise (e.g., reducing model size to fit 16GB VRAM), explicitly explain the trade-off.
*   When debugging errors, provide the diagnostic steps first, followed by the solution.

