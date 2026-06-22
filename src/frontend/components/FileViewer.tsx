"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { Search, FileText, FileType, Image as ImageIcon, Folder, ChevronRight } from "lucide-react"
import { fileViewerRawUrl, loadFileViewerText } from "@/lib/api"
import { LaTeXMarkdown } from "@/components/LaTeXMarkdown"
import { FloatingWindow } from "@/components/FloatingWindow"
import { fuzzyScore } from "@/lib/fuzzy"
import { buildPathTree, relativeName, type PathTreeNode } from "@/lib/pathTree"
import { cn } from "@/lib/utils"

const PdfViewer = dynamic(() => import("./PdfViewer").then((m) => ({ default: m.PdfViewer })), { ssr: false })

const RESULT_CAP = 300
const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "avif", "ico"])

type FileKind = "image" | "pdf" | "markdown" | "text"

function baseName(path: string): string {
  const i = path.lastIndexOf("/")
  return i === -1 ? path : path.slice(i + 1)
}

function dirName(path: string): string {
  const i = path.lastIndexOf("/")
  return i === -1 ? "" : path.slice(0, i)
}

function extOf(path: string): string {
  const name = baseName(path)
  const i = name.lastIndexOf(".")
  return i <= 0 ? "" : name.slice(i + 1).toLowerCase()
}

function kindOf(path: string): FileKind {
  const ext = extOf(path)
  if (IMAGE_EXTS.has(ext)) return "image"
  if (ext === "pdf") return "pdf"
  if (ext === "md" || ext === "markdown") return "markdown"
  return "text"
}

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
  /** Controls rendered at the right of the search bar (e.g. an upload button). */
  headerActions?: React.ReactNode
  /** Per-row trailing control (e.g. an add-to-project button). */
  renderRowSuffix?: (path: string) => React.ReactNode
  onArrowLeft?: () => void
  onArrowRight?: () => void
}

export function FileViewer({
  files,
  onSelect,
  onClose,
  placeholder = "Search files…",
  emptyLabel = "No files",
  statusLabel,
  headerActions,
  renderRowSuffix,
  onArrowLeft,
  onArrowRight,
}: FileViewerProps) {
  const [query, setQuery] = useState("")
  const [idx, setIdx] = useState(0)
  const [preview, setPreview] = useState<string>("")
  const cache = useRef<Map<string, string>>(new Map())
  const listRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const searching = query.trim().length > 0

  // Group filepaths into a chain-collapsed directory tree (built once per list).
  const tree = useMemo(() => buildPathTree(files), [files])
  const [stack, setStack] = useState<PathTreeNode[]>([tree])
  useEffect(() => { setStack([tree]) }, [tree])
  const currentNode = stack[stack.length - 1] ?? tree

  const entries = useMemo<Row[]>(() => {
    const q = query.trim()
    if (q) {
      // Search ignores the directory context — flat fuzzy across every file.
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

  useEffect(() => { setIdx(0) }, [query, currentNode])
  useEffect(() => { inputRef.current?.focus() }, [])

  const highlighted = entries[idx]
  const highlightedKind = highlighted?.type === "file" ? kindOf(highlighted.path) : null

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
  useEffect(() => {
    if (highlighted?.type !== "file" || (highlightedKind !== "markdown" && highlightedKind !== "text")) {
      setPreview("")
      return
    }
    const path = highlighted.path
    const cached = cache.current.get(path)
    if (cached !== undefined) { setPreview(cached); return }
    let cancelled = false
    setPreview("")
    const t = setTimeout(() => {
      loadFileViewerText(path)
        .then((content) => { if (!cancelled) { cache.current.set(path, content); setPreview(content) } })
        .catch(() => { if (!cancelled) setPreview("") })
    }, 110)
    return () => { cancelled = true; clearTimeout(t) }
  }, [highlighted, highlightedKind])

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
            {headerActions}
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
            {highlighted?.type === "file" ? highlighted.path : "Select a file to preview"}
          </div>
          <div className={cn("min-h-0 flex-1", highlightedKind === "pdf" ? "" : "overflow-y-auto p-4")}>
            {highlighted?.type === "dir" ? (
              <div className="text-xs text-ink-faint">Folder — press → or Enter to open</div>
            ) : highlighted?.type !== "file" ? (
              <div className="text-xs text-ink-faint" />
            ) : highlightedKind === "image" ? (
              <img src={fileViewerRawUrl(highlighted.path)} alt={highlighted.label} className="mx-auto max-h-full max-w-full rounded border border-divider object-contain" />
            ) : highlightedKind === "pdf" ? (
              <PdfViewer url={fileViewerRawUrl(highlighted.path)} />
            ) : preview ? (
              highlightedKind === "markdown" ? (
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
