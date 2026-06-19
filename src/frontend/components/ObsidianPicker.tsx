"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Search, Folder, FileText, ChevronRight, CornerDownLeft } from "lucide-react"
import type { ObsidianFile } from "@/lib/types"
import { readObsidianFile } from "@/lib/api"
import { LaTeXMarkdown } from "@/components/LaTeXMarkdown"
import { FloatingWindow } from "@/components/FloatingWindow"
import { cn } from "@/lib/utils"

type Entry = { type: "dir" | "file"; name: string; path: string }

const RESULT_CAP = 200

function parentDir(dir: string): string {
  const i = dir.lastIndexOf("/")
  return i === -1 ? "" : dir.slice(0, i)
}

function browseEntries(files: ObsidianFile[], currentDir: string): Entry[] {
  const prefix = currentDir ? `${currentDir}/` : ""
  const dirs = new Set<string>()
  const fileEntries: Entry[] = []
  for (const f of files) {
    if (!f.path.startsWith(prefix)) continue
    const rest = f.path.slice(prefix.length)
    const slash = rest.indexOf("/")
    if (slash === -1) fileEntries.push({ type: "file", name: rest, path: f.path })
    else dirs.add(rest.slice(0, slash))
  }
  const dirEntries: Entry[] = [...dirs]
    .sort((a, b) => a.localeCompare(b))
    .map((d) => ({ type: "dir", name: d, path: prefix + d }))
  fileEntries.sort((a, b) => a.name.localeCompare(b.name))
  return [...dirEntries, ...fileEntries]
}

function fuzzyScore(text: string, query: string): number | null {
  const t = text.toLowerCase()
  const q = query.toLowerCase()
  let ti = 0
  let score = 0
  let prev = -2
  for (const ch of q) {
    let found = -1
    for (let k = ti; k < t.length; k++) {
      if (t[k] === ch) { found = k; break }
    }
    if (found === -1) return null
    score += found === prev + 1 ? 3 : 1
    const before = found === 0 ? "/" : t[found - 1]
    if (before === "/" || before === " " || before === "-" || before === "_") score += 2
    prev = found
    ti = found + 1
  }
  return score - text.length * 0.01
}

interface ObsidianPickerProps {
  files: ObsidianFile[]
  onSelect: (path: string) => void
  onClose: () => void
}

export function ObsidianPicker({ files, onSelect, onClose }: ObsidianPickerProps) {
  const [query, setQuery] = useState("")
  const [currentDir, setCurrentDir] = useState("")
  const [idx, setIdx] = useState(0)
  const [preview, setPreview] = useState<string>("")
  const cache = useRef<Map<string, string>>(new Map())
  const listRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const searching = query.trim().length > 0

  const entries = useMemo<Entry[]>(() => {
    const q = query.trim()
    if (!q) return browseEntries(files, currentDir)
    return files
      .map((f) => {
        const byPath = fuzzyScore(f.path, q)
        const byName = fuzzyScore(f.name, q)
        const s = Math.max(byPath ?? -Infinity, byName === null ? -Infinity : byName + 1)
        return { f, s }
      })
      .filter((x) => x.s > -Infinity)
      .sort((a, b) => b.s - a.s)
      .slice(0, RESULT_CAP)
      .map((x) => ({ type: "file" as const, name: x.f.name, path: x.f.path }))
  }, [files, query, currentDir])

  useEffect(() => { setIdx(0) }, [query, currentDir])
  useEffect(() => { inputRef.current?.focus() }, [])

  const highlighted = entries[idx]

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${idx}"]`)
    el?.scrollIntoView({ block: "nearest" })
  }, [idx])

  useEffect(() => {
    if (!highlighted || highlighted.type !== "file") { setPreview(""); return }
    const path = highlighted.path
    const cached = cache.current.get(path)
    if (cached !== undefined) { setPreview(cached); return }
    let cancelled = false
    setPreview("")
    const t = setTimeout(() => {
      readObsidianFile(path)
        .then((r) => { if (!cancelled) { cache.current.set(path, r.content); setPreview(r.content) } })
        .catch(() => { if (!cancelled) setPreview("") })
    }, 110)
    return () => { cancelled = true; clearTimeout(t) }
  }, [highlighted?.path, highlighted?.type])

  const descend = useCallback((entry: Entry | undefined) => {
    if (!entry) return
    if (entry.type === "dir") { setQuery(""); setCurrentDir(entry.path) }
    else onSelect(entry.path)
  }, [onSelect])

  const ascend = useCallback(() => {
    if (currentDir) setCurrentDir((d) => parentDir(d))
  }, [currentDir])

  const onKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault(); setIdx((i) => Math.min(i + 1, entries.length - 1)); break
      case "ArrowUp":
        e.preventDefault(); setIdx((i) => Math.max(i - 1, 0)); break
      case "ArrowRight":
        if (!searching) { e.preventDefault(); descend(entries[idx]) }
        break
      case "ArrowLeft":
        if (!searching) { e.preventDefault(); ascend() }
        break
      case "Enter":
        e.preventDefault(); descend(entries[idx]); break
      case "Backspace":
        if (!searching && currentDir) { e.preventDefault(); ascend() }
        break
    }
  }, [entries, idx, searching, currentDir, descend, ascend])

  return (
    <FloatingWindow onClose={onClose} panelClassName="h-[min(560px,85vh)] max-w-4xl">
      <div className="flex min-h-0 flex-1">
        {/* Left: search + navigable list */}
        <div className="flex w-[42%] min-w-0 flex-col border-r border-divider">
          <div className="flex items-center gap-2 border-b border-divider px-3 py-2.5">
            <Search className="h-3.5 w-3.5 shrink-0 text-ink-subtle" />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Search vault…"
              spellCheck={false}
              autoComplete="off"
              className="w-full bg-transparent text-sm text-ink placeholder:text-ink-subtle outline-none"
            />
          </div>

          {!searching && (
            <div className="flex items-center gap-1 border-b border-divider/50 px-3 py-1.5 text-[10px] uppercase tracking-wider text-ink-faint">
              <Folder className="h-3 w-3 shrink-0" />
              <span className="truncate">{currentDir || "vault root"}</span>
            </div>
          )}

          <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto py-1">
            {entries.length === 0 && (
              <div className="px-3 py-6 text-center text-xs text-ink-faint">No notes</div>
            )}
            {entries.map((entry, i) => {
              const selected = i === idx
              const Icon = entry.type === "dir" ? Folder : FileText
              return (
                <button
                  key={entry.path}
                  data-idx={i}
                  type="button"
                  onMouseEnter={() => setIdx(i)}
                  onClick={() => descend(entry)}
                  className={cn(
                    "flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors",
                    selected ? "bg-surface-elevated text-ink" : "text-ink-muted hover:bg-hover",
                  )}
                >
                  <Icon className={cn("h-3.5 w-3.5 shrink-0", entry.type === "dir" ? "text-ink-subtle" : "text-ink-faint")} />
                  {searching ? (
                    <span className="min-w-0 flex-1 truncate">
                      <span className="text-ink-faint">{entry.path.includes("/") ? entry.path.slice(0, entry.path.lastIndexOf("/") + 1) : ""}</span>
                      <span>{entry.name}</span>
                    </span>
                  ) : (
                    <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                  )}
                  {entry.type === "dir" && <ChevronRight className="h-3 w-3 shrink-0 text-ink-faint" />}
                </button>
              )
            })}
          </div>

          <div className="flex items-center gap-3 border-t border-divider px-3 py-2 text-[10px] text-ink-faint">
            <span>↑↓ move</span>
            {!searching && <span>→ open</span>}
            {!searching && <span>← up</span>}
            <span className="flex items-center gap-1"><CornerDownLeft className="h-2.5 w-2.5" /> attach</span>
            <span className="ml-auto">esc</span>
          </div>
        </div>

        {/* Right: scrollable markdown preview */}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="truncate border-b border-divider px-4 py-2.5 text-[11px] text-ink-subtle">
            {highlighted?.type === "file" ? highlighted.path : "Select a note to preview"}
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {highlighted?.type === "file" ? (
              preview ? (
                <LaTeXMarkdown content={preview} />
              ) : (
                <div className="text-xs text-ink-faint">Loading preview…</div>
              )
            ) : (
              <div className="text-xs text-ink-faint">{highlighted?.type === "dir" ? "Folder — press → or Enter to open" : ""}</div>
            )}
          </div>
        </div>
      </div>
    </FloatingWindow>
  )
}
