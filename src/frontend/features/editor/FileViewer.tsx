"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Search, FileText, FileType, Image as ImageIcon, Folder, ChevronRight } from "lucide-react"
import { loadFileViewerText } from "@/lib/backend/api"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { fuzzyScore } from "@/lib/fuzzy"
import { baseName, dirName, extOf, kindOf, type FileKind } from "@/lib/paths"
import { buildPathTree, relativeName, type PathTreeNode } from "@/lib/pathTree"
import { cn } from "@/lib/utils"

import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"

// Max rows rendered for a search.
const RESULT_CAP = 300

function iconFor(kind: FileKind) {
  if (kind === "image") return ImageIcon
  if (kind === "pdf") return FileType
  return FileText
}

function sortFiles(paths: string[]): string[] {
  return [...paths].sort((a, b) => {
    const ea = extOf(a)
    const eb = extOf(b)
    if (ea !== eb) return ea.localeCompare(eb)
    return baseName(a).toLowerCase().localeCompare(baseName(b).toLowerCase())
  })
}

type Row =
  | { type: "dir"; label: string; path: string; node: PathTreeNode }
  | { type: "file"; label: string; path: string }

interface FileViewerProps {
  files: string[]
  onSelect: (path: string) => void
  onClose: () => void
  placeholder?: string
  emptyLabel?: string
  /** Scope/location row shown above the list (e.g. picker scope indicator). */
  statusLabel?: React.ReactNode
  /** Per-row trailing control (e.g. an add-to-project button). */
  renderRowSuffix?: (path: string) => React.ReactNode
  onArrowLeft?: () => void
  onArrowRight?: () => void
}

/** Floating file picker with fuzzy search, directory drill-down and a type-aware preview pane. */
export function FileViewer({
  files,
  onSelect,
  onClose,
  placeholder = "Search files…",
  emptyLabel = "No files",
  statusLabel,
  renderRowSuffix,
  onArrowLeft,
  onArrowRight,
}: FileViewerProps) {
  const [query, setQuery] = useState("")
  const [idx, setIdx] = useState(0)
  const [loaded, setLoaded] = useState<{ path: string; content: string } | null>(null)
  const cache = useRef<Map<string, string>>(new Map())
  const listRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const searching = query.trim().length > 0

  // Group filepaths into a chain-collapsed directory tree (built once per list).
  const tree = useMemo(() => buildPathTree(files), [files])
  const [stack, setStack] = useState<PathTreeNode[]>([tree])
  const [stackTree, setStackTree] = useState(tree)
  // Return to the root when the file list changes
  if (stackTree !== tree) {
    setStackTree(tree)
    setStack([tree])
  }
  const currentNode = stack[stack.length - 1] ?? tree

  const entries = useMemo<Row[]>(() => {
    const q = query.trim()
    if (q) {
      // Search ignores the directory context: flat fuzzy across every file, name matches get a +1 bonus.
      return sortFiles(files)
        .map((path) => {
          const byName = fuzzyScore(baseName(path), q)
          const byPath = fuzzyScore(path, q)
          const s = Math.max(byName === null ? -Infinity : byName + 1, byPath ?? -Infinity)
          return { path, s }
        })
        .filter((x) => x.s > -Infinity)
        .sort((a, b) => b.s - a.s)
        .slice(0, RESULT_CAP)
        .map((x) => ({ type: "file" as const, label: baseName(x.path), path: x.path }))
    }
    const dirRows: Row[] = [...currentNode.dirs]
      .map((node) => ({ type: "dir" as const, label: relativeName(currentNode.path, node.path), path: node.path, node }))
      .sort((a, b) => a.label.toLowerCase().localeCompare(b.label.toLowerCase()))
    const fileRows: Row[] = sortFiles(currentNode.files).map((path) => ({ type: "file" as const, label: baseName(path), path }))
    return [...dirRows, ...fileRows]
  }, [files, query, currentNode])

  const [idxScope, setIdxScope] = useState({ query, currentNode })
  // Highlight the first row whenever the search or directory changes
  if (idxScope.query !== query || idxScope.currentNode !== currentNode) {
    setIdxScope({ query, currentNode })
    setIdx(0)
  }
  useEffect(() => { inputRef.current?.focus() }, [])

  const highlighted = entries[idx]

  // Debounce the previewed row separately from the highlighted one. Hovering fast changes idx on
  // every mouseenter, and swapping PdfViewer's url that quickly tears down its pdf.js transport mid-render.
  const [previewed, setPreviewed] = useState<Row | undefined>(undefined)
  useEffect(() => {
    const t = setTimeout(() => setPreviewed(highlighted), 120)
    return () => clearTimeout(t)
  }, [highlighted])
  const previewedKind = previewed?.type === "file" ? kindOf(previewed.path) : null

  const breadcrumb = useMemo(
    () => stack.map((n, i) => (i === 0 ? baseName(n.path) || n.path || "root" : relativeName(stack[i - 1].path, n.path))).join(" / "),
    [stack],
  )

  const enterDir = useCallback((node: PathTreeNode) => {
    setQuery("")
    setStack((s) => [...s, node])
  }, [])

  const goUp = useCallback(() => {
    setStack((s) => (s.length > 1 ? s.slice(0, -1) : s))
  }, [])

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${idx}"]`)
    el?.scrollIntoView({ block: "nearest" })
  }, [idx])

  // Text/markdown previews are fetched + cached; images/pdfs render from a URL.
  const textPath = previewed?.type === "file" && (previewedKind === "markdown" || previewedKind === "text") ? previewed.path : null
  useEffect(() => {
    if (textPath === null) return
    let cancelled = false
    const cached = cache.current.get(textPath)
    const request = cached !== undefined ? Promise.resolve(cached) : loadFileViewerText(textPath)
    request
      .then((content) => { if (!cancelled) { cache.current.set(textPath, content); setLoaded({ path: textPath, content }) } })
      .catch(() => { if (!cancelled) setLoaded({ path: textPath, content: "" }) })
    return () => { cancelled = true }
  }, [textPath])
  const preview = loaded?.path === textPath ? loaded.content : ""

  const activate = useCallback((row: Row | undefined) => {
    if (!row) return
    if (row.type === "dir") enterDir(row.node)
    else onSelect(row.path)
  }, [enterDir, onSelect])

  const onKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault(); setIdx((i) => Math.min(i + 1, entries.length - 1)); break
      case "ArrowUp":
        e.preventDefault(); setIdx((i) => Math.max(i - 1, 0)); break
      case "ArrowLeft":
        // Caret stays free while searching; otherwise climb the directory tree,
        // delegating to the consumer only when already at the top.
        if (searching) break
        if (stack.length > 1) { e.preventDefault(); goUp() }
        else if (onArrowLeft) { e.preventDefault(); onArrowLeft() }
        break
      case "ArrowRight": {
        if (searching) break
        const row = entries[idx]
        if (row?.type === "dir") { e.preventDefault(); enterDir(row.node) }
        else if (onArrowRight) { e.preventDefault(); onArrowRight() }
        break
      }
      case "Enter":
        e.preventDefault(); activate(entries[idx]); break
    }
  }, [entries, idx, searching, stack.length, goUp, enterDir, activate, onArrowLeft, onArrowRight])

  return (
    <FloatingWindow onClose={onClose} panelClassName="h-[86vh] max-h-[840px]">
      <div className="flex min-h-0 flex-1">
        {/* Left: navigable list with ghost search */}
        <div className="flex w-[42%] min-w-0 flex-col border-r border-divider/30" onClick={() => inputRef.current?.focus()}>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            spellCheck={false}
            autoComplete="off"
            className="sr-only"
            aria-label={placeholder}
          />

          <div className="flex items-center justify-between border-b border-divider/30 px-3 py-2">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-ink-faint min-w-0">
              {searching ? (
                <><Search className="h-3 w-3 shrink-0" /><span className="truncate">{query}</span></>
              ) : statusLabel ? (
                statusLabel
              ) : (
                <><Folder className="h-3 w-3 shrink-0" /><span className="truncate">{breadcrumb}</span></>
              )}
            </div>
          </div>

          <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto py-1">
            {entries.length === 0 && (
              <div className="px-3 py-6 text-center text-xs text-ink-faint">{emptyLabel}</div>
            )}
            {entries.map((row, i) => {
              const selected = i === idx
              const Icon = row.type === "dir" ? Folder : iconFor(kindOf(row.path))
              const dir = row.type === "file" && searching ? dirName(row.path) : ""
              return (
                <button
                  key={row.path}
                  data-idx={i}
                  type="button"
                  onMouseEnter={() => setIdx(i)}
                  onClick={() => activate(row)}
                  className={cn(
                    "group flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors",
                    selected ? "bg-surface-elevated text-ink" : "text-ink-muted hover:bg-hover",
                  )}
                >
                  <Icon className={cn("h-3.5 w-3.5 shrink-0", row.type === "dir" ? "text-ink-subtle" : "text-ink-faint")} />
                  <span className="min-w-0 flex-1 truncate">{row.label}</span>
                  {row.type === "file" && renderRowSuffix?.(row.path)}
                  {dir && <span className="max-w-[40%] truncate text-[10px] text-ink-faint">{dir}</span>}
                  {row.type === "dir" && <ChevronRight className="h-3 w-3 shrink-0 text-ink-faint" />}
                </button>
              )
            })}
          </div>
        </div>

        {/* Right: type-aware preview */}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="truncate border-b border-divider/30 px-4 py-2 text-[11px] text-ink-subtle">
            {previewed?.type === "file" ? previewed.path : "Select a file to preview"}
          </div>
          <div className={cn("min-h-0 flex-1", previewedKind === "pdf" ? "" : "overflow-y-auto p-4")}>
            {previewed?.type === "dir" ? (
              <div className="text-xs text-ink-faint">Folder — press → or Enter to open</div>
            ) : previewed?.type !== "file" ? (
              <div className="text-xs text-ink-faint" />
            ) : previewedKind === "image" ? (
              // eslint-disable-next-line @next/next/no-img-element -- dynamic backend raw-file URL in a Tauri static export, next/image cannot optimize it
              <img src={fileViewerRawUrl(previewed.path)} alt={previewed.label} className="mx-auto max-h-full max-w-full rounded border border-divider object-contain" />
            ) : previewedKind === "pdf" ? (
              <LazyPdfViewer key={previewed.path} url={fileViewerRawUrl(previewed.path)} />
            ) : preview ? (
              previewedKind === "markdown" ? (
                <LaTeXMarkdown content={preview} />
              ) : (
                <pre className="whitespace-pre-wrap break-words font-mono text-[11px] text-ink-muted">{preview}</pre>
              )
            ) : (
              <div className="text-xs text-ink-faint">Loading preview…</div>
            )}
          </div>
        </div>
      </div>
    </FloatingWindow>
  )
}
