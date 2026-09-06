# System Overview

## What runs where

GigaChad Bot is a local-first, single-user LLM workspace: a Next.js/React
client (`:2999`) talks to a FastAPI backend (`:8001`) which owns all
persistence and filesystem access. The backend is the sole authority for
paths — the browser only ever sees paths the server already issued.

`./run.sh` starts both. On Linux desktop, the frontend is statically exported
into a Tauri webview and the backend is frozen into a PyInstaller sidecar that
Tauri launches on an ephemeral loopback port; the browser-facing API surface
(FastAPI) is unchanged, Tauri just supplies the base URL after a readiness
handshake.

## External services it integrates with

The backend is a coordinator over several optional local/cloud services —
none of them are reimplemented in-repo:

- **LiteLLM** — the model-provider abstraction. All chat/completions go
  through it, so Gemini, DeepSeek, OpenRouter, and local providers are
  swappable behind one interface (`src/config.py` holds defaults).
- **Vane** (`VANE_URL`, default `localhost:3001`) — a local sidecar that
  performs web search on the backend's behalf. `routes/search.py` proxies to
  it, resolving the user-facing model id to a Vane provider/model id and
  caching that resolution. Vane itself is backed by a **SearXNG** instance
  (`SEARX_URL`); DeepSeek is routed through `vane-deepseek-shim.py` because
  DeepSeek rejects Vane's `json_schema` response format (see
  `vane-setup.md`).
- **SearXNG** — also used directly by **GPT-Researcher**, which powers
  `/api/research` (streams progress/result over SSE, persists JSON traces via
  `research_trace.py`).
- **MinerU** — PDF parsing/OCR. `routes/mineru.py` + `ExtractQueue`
  (`extract_queue.py`) manage the parse lifecycle; parsed Markdown backs both
  the "Study" mode (mind map/overview/article generation) and the document
  library's PDF preview. The desktop sidecar bundles only MinerU's HTTP
  *client* — the torch/CUDA OCR server is excluded, so desktop OCR requires an
  external `mineru.cli.fast_api` instance via `MINERU_SERVER_URL`
  (`remote-inference.md`).
- **Nextcloud** (or any configured cloud-synced directory) — holds the global
  document library; user-authored Markdown/LaTeX/drawings are mirrored there
  from `chat_histories`.
- **Tauri** — the desktop shell only; it has no filesystem authority of its
  own and exists purely to host the exported UI + sidecar.

All of the above are optional/user-configured — the app degrades gracefully
(e.g. web search and OCR just aren't available) when they're not running.

## Memory system

Memory is a **document-backed, LLM-mediated** store (`src/lib/memory_store.py`,
`routes/memory.py`) — no vector DB. It has two independent scopes:

- **global** — durable facts about the user, valid across all projects
  (identity, communication/learning/engineering preferences, goals).
- **project** — facts about one active project (purpose, current focus, key
  concepts, decisions, constraints, resources, open questions).

Each scope has its own configurable set of **categories**
(`global-categories.json` / a project's `memory/categories.json`, falling back
to built-in defaults); every stored memory belongs to exactly one category.

**Pipeline**, all user-gated:

1. **Extract** — a small model (`MEMORY_MODEL`) reads the conversation tail
   since that chat's last memorization and emits only *new or corrected*
   atomic facts as JSON, given the categories and what's already on record (it
   never re-proposes known facts). Candidates are buffered to
   `memory/pending/<review_id>.json`. Global and project extraction run
   concurrently and independently.
2. **Review** — the frontend shows the candidates; the user accepts, rejects,
   or edits them. Nothing is written to the canonical store yet.
3. **Reconcile** — accepted candidates are merged into the canonical list
   *per category only* (a category with no new candidates is untouched); an
   LLM call merges overlapping/contradictory text within that one category
   and returns the deduplicated canonical list, tagging each result
   `pre-existing` / `new` / `combined` for the diff view (`preview`/`commit`
   share this path).
4. **Commit** — the canonical JSON (`global-profile.json` or a project's
   `memory/memory.json`) and a rendered Markdown doc
   (`global-profile.md` / `memory/memory.md`) are both written, and a
   per-`(chat_id, scope)` **watermark** advances so the same messages aren't
   re-extracted later. Cancelling discards the pending buffer without moving
   the watermark.

At chat time, `MemoryStore.augment_system_prompt` reads both Markdown docs
(global profile + the active project's memory, if any) and appends them to
the system prompt under a `# Persistent Memory Context` header — this is how
the LLM actually "remembers" across chats. Users can also list, move a memory
between scopes, or re-bucket memories orphaned by a category rename/deletion
(`remap_orphaned`), all without touching the raw JSON by hand.
