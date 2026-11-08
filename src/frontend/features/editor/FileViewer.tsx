"use client"

import { useEffect, useMemo, useState } from "react"
import { FileText, FileType, Image as ImageIcon, Folder, ChevronRight } from "lucide-react"
import { loadFileViewerText } from "@/lib/backend/api"
import { RawFileView } from "./RawFileView"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { FloatingPicker } from "@/components/ui/CommandPicker"
import { fuzzyPathScore, rankByScore } from "@/lib/fuzzy"
import { baseName, dirName, extOf, kindOf, type FileKind } from "@/lib/paths"
import { buildPathTree, relativeName, type PathTreeNode } from "@/lib/pathTree"
import { cn } from "@/lib/utils"
import { useResetOnChange } from "@/hooks/useResetOnChange"

// Max rows rendered for a search.
const RESULT_CAP = 300
/** Hovering fast changes the highlight on every row, swapping PdfViewer's url that quickly tears down its pdf.js transport mid-render. */
const PREVIEW_DELAY_MS = 120

function iconFor(kind: FileKind) {
  if (kind === "image") return ImageIcon
  if (kind === "pdf") return FileType
  return FileText
}

function sortFiles(paths: string[], nameOf: (path: string) => string): string[] {
  return [...paths].sort((a, b) => {
    const ea = extOf(a)
    const eb = extOf(b)
    if (ea !== eb) return ea.localeCompare(eb)
    return nameOf(a).toLowerCase().localeCompare(nameOf(b).toLowerCase())
  })
}

type Row =
  | { type: "dir"; label: string; path: string; node: PathTreeNode }
  | { type: "file"; label: string; path: string }

/** The file at `path` as it opens: images and PDFs from their URL, Markdown rendered, other text as is. Texts are fetched once into `cache`. */
function FilePreview({ path, cache }: { path: string; cache: Map<string, string> }) {
  const [shownPath, setShownPath] = useState<string | null>(null)
  useEffect(() => {
    const timer = setTimeout(() => setShownPath(path), PREVIEW_DELAY_MS)
    return () => clearTimeout(timer)
  }, [path])
  const kind = shownPath ? kindOf(shownPath) : null
  const textPath = shownPath && (kind === "markdown" || kind === "text") ? shownPath : null
  const [loaded, setLoaded] = useState<{ path: string; content: string } | null>(null)
  useEffect(() => {
    if (textPath === null) return
    let cancelled = false
    const cached = cache.get(textPath)
    const request = cached !== undefined ? Promise.resolve(cached) : loadFileViewerText(textPath)
    request
      .then((content) => { if (!cancelled) { cache.set(textPath, content); setLoaded({ path: textPath, content }) } })
      .catch(() => { if (!cancelled) setLoaded({ path: textPath, content: "" }) })
    return () => { cancelled = true }
  }, [textPath, cache])
  const text = loaded?.path === textPath ? loaded.content : null

  return <div className="flex h-full min-h-0 flex-col">
    <div className="truncate border-b border-divider/30 px-4 py-2 text-[11px] text-ink-subtle">{path}</div>
    {shownPath && (kind === "image" || kind === "pdf") ? (
      <RawFileView path={shownPath} kind={kind} />
    ) : (
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {text === null ? <div className="text-xs text-ink-faint">Loading preview…</div>
          : kind === "markdown" ? <LaTeXMarkdown content={text} />
          : <pre className="whitespace-pre-wrap break-words font-mono text-[11px] text-ink-muted">{text}</pre>}
      </div>
    )}
  </div>
}

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
  /** Name a file row shows, its file name without. */
  nameOf?: (path: string) => string
}

/** File picker with fuzzy search, directory drill-down (ArrowRight in, ArrowLeft up) and the highlighted file previewed in the picker's window. */
export function FileViewer({
  files,
  onSelect,
  onClose,
  placeholder = "Search files…",
  emptyLabel = "No files",
  statusLabel,
  renderRowSuffix,
  onArrowLeft,
  nameOf = baseName,
}: FileViewerProps) {
  const [query, setQuery] = useState("")
  const [cache] = useState(() => new Map<string, string>())
  const searching = query.trim().length > 0

  // Group filepaths into a chain-collapsed directory tree (built once per list).
  const tree = useMemo(() => buildPathTree(files), [files])
  const [stack, setStack] = useState<PathTreeNode[]>([tree])
  // Return to the root when the file list changes
  useResetOnChange(tree, (next) => setStack([next]))
  const currentNode = stack[stack.length - 1] ?? tree

  const rows = useMemo<Row[]>(() => {
    const q = query.trim()
    if (q) {
      // Search ignores the directory context: flat fuzzy across every file, name matches rank first
      return rankByScore(sortFiles(files, nameOf), (path) => fuzzyPathScore(path, q, nameOf(path)))
        .slice(0, RESULT_CAP)
        .map((path) => ({ type: "file" as const, label: nameOf(path), path }))
    }
    const dirRows: Row[] = [...currentNode.dirs]
      .map((node) => ({ type: "dir" as const, label: relativeName(currentNode.path, node.path), path: node.path, node }))
      .sort((a, b) => a.label.toLowerCase().localeCompare(b.label.toLowerCase()))
    const fileRows: Row[] = sortFiles(currentNode.files, nameOf).map((path) => ({ type: "file" as const, label: nameOf(path), path }))
    return [...dirRows, ...fileRows]
  }, [files, query, currentNode, nameOf])

  const breadcrumb = stack.map((n, i) => (i === 0 ? baseName(n.path) || n.path || "root" : relativeName(stack[i - 1].path, n.path))).join(" / ")
  const enterDir = (node: PathTreeNode) => {
    setQuery("")
    setStack((s) => [...s, node])
  }

  return <FloatingPicker<Row>
    label={placeholder}
    groups={[{ key: "rows", items: rows }]}
    itemKey={(row) => `${row.type}:${row.path}`}
    searchPlaceholder={placeholder}
    query={query}
    onQueryChange={setQuery}
    emptyText={emptyLabel}
    header={searching ? statusLabel : statusLabel ?? <><Folder className="h-3 w-3 shrink-0" /><span className="truncate">{breadcrumb}</span></>}
    renderItem={(row) => {
      const Icon = row.type === "dir" ? Folder : iconFor(kindOf(row.path))
      const dir = row.type === "file" && searching ? dirName(row.path) : ""
      return <>
        <Icon className={cn("h-3.5 w-3.5 shrink-0", row.type === "dir" ? "text-ink-subtle" : "text-ink-faint")} />
        <span className="min-w-0 flex-1 truncate">{row.label}</span>
        {row.type === "file" && renderRowSuffix?.(row.path)}
        {dir && <span className="max-w-[40%] truncate text-[10px] text-ink-faint">{dir}</span>}
        {row.type === "dir" && <ChevronRight className="h-3 w-3 shrink-0 text-ink-faint" />}
      </>
    }}
    renderPreview={(row) => row.type === "file" ? <FilePreview path={row.path} cache={cache} /> : null}
    onKeyDown={(event, row) => {
      // The caret stays free while searching, otherwise the arrows walk the folder tree
      if (searching) return false
      if (event.key === "ArrowLeft") {
        if (stack.length > 1) setStack((s) => s.slice(0, -1))
        else if (onArrowLeft) onArrowLeft()
        else return false
      } else if (event.key === "ArrowRight" && row?.type === "dir") enterDir(row.node)
      else return false
      event.preventDefault()
      return true
    }}
    onPick={(row) => {
      if (row.type === "file") {
        onSelect(row.path)
        return true
      }
      enterDir(row.node)
      return false
    }}
    onClose={onClose}
  />
}
