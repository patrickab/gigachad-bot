"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { useFileTree } from "@/components/ui/file-tree/useFileTree"
import { fetchRootDiffFiles, fetchRootFileDiff, fetchRootDiffContent, writeRootFile, type DiffStatus, type RootDiffFiles } from "@/lib/backend/api"
import { parsePatch, type PatchLine } from "@/lib/diff"
import { baseName } from "@/lib/paths"
import { cn } from "@/lib/utils"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, useNeovimTerminal, type NeovimDocument } from "@/features/editor/NeovimEditor"

type FilePatch = { lines: PatchLine[] | null; failed: boolean }

const STATUS_BADGE: Record<DiffStatus, string> = { modified: "M", added: "A", deleted: "D", renamed: "R", untracked: "U", conflict: "C" }

const LINE_CLASS: Record<PatchLine["kind"], string> = {
  hunk: "bg-surface text-ink-faint",
  added: "bg-diff-added/10 text-diff-added",
  removed: "bg-diff-removed/10 text-diff-removed",
  context: "text-ink-muted",
  note: "text-ink-faint italic",
}

/** Nest changed files under their directories, folder ids are `dir:<path>`. */
function buildTree(files: RootDiffFiles["files"], patches: Record<string, FilePatch>, activePath: string | null, select: (path: string) => void): FileTreeItem<string>[] {
  const roots: FileTreeItem<string>[] = []
  for (const file of files) {
    const parts = file.path.split("/")
    let siblings = roots
    parts.slice(0, -1).forEach((name, depth) => {
      const id = `dir:${parts.slice(0, depth + 1).join("/")}`
      let folder = siblings.find((item) => item.id === id)
      if (!folder) {
        folder = { id, label: name, type: "folder", children: [] }
        siblings.push(folder)
      }
      siblings = folder.children!
    })
    const lines = patches[file.path]?.lines
    siblings.push({
      id: file.path, label: parts[parts.length - 1], type: "element", badge: STATUS_BADGE[file.status],
      meta: lines && <LineCountLabel counts={countLines(lines)} />,
      isActive: file.path === activePath, onSelect: () => select(file.path),
    })
  }
  return compactFolders(roots)
}

/** Merge chains of folders holding only one folder into a single `a/b/c` row, the row keeps the deepest id. */
function compactFolders(items: FileTreeItem<string>[]): FileTreeItem<string>[] {
  return items.map((item) => {
    let folder = item
    while (folder.children?.length === 1 && folder.children[0]!.type === "folder") {
      const child = folder.children[0]!
      folder = { ...child, label: `${folder.label}/${child.label}` }
    }
    return folder.children ? { ...folder, children: compactFolders(folder.children) } : folder
  })
}

function folderIds(files: RootDiffFiles["files"]): string[] {
  return files.flatMap((file) => file.path.split("/").slice(0, -1).map((_, depth, parts) => `dir:${parts.slice(0, depth + 1).join("/")}`))
}

function ChangedFiles({ files, patches, activePath, onSelect, root }: {
  files: RootDiffFiles["files"]; patches: Record<string, FilePatch>; activePath: string | null; onSelect: (path: string) => void; root: string
}) {
  const controller = useFileTree({ defaultExpanded: folderIds(files) })
  return <FileTree label={`Changes in ${baseName(root)}`} items={buildTree(files, patches, activePath, onSelect)} controller={controller} />
}

type LineCounts = { added: number; removed: number }

const totalsCache = new Map<string, LineCounts>()

function countLines(lines: PatchLine[]): LineCounts {
  return {
    added: lines.filter((line) => line.kind === "added").length,
    removed: lines.filter((line) => line.kind === "removed").length,
  }
}

function LineCountLabel({ counts }: { counts: LineCounts }) {
  return <>
    <span className="text-diff-added">+{counts.added}</span>
    <span className="text-diff-removed">−{counts.removed}</span>
  </>
}

/** Pull-request style totals: changed files, then added and removed lines. Shown only once every patch loaded, else the last known totals of this root. */
function ChangeSummary({ root, files, patches }: { root: string; files: number; patches: FilePatch[] }) {
  if (patches.length === files) totalsCache.set(root, countLines(patches.flatMap((patch) => patch.lines ?? [])))
  const totals = totalsCache.get(root)
  return <p aria-label="Change summary" className="flex items-center gap-2 px-2 pb-2 text-xs tabular-nums text-ink-faint">
    <span>{files} {files === 1 ? "file" : "files"}</span>
    {totals && <LineCountLabel counts={totals} />}
  </p>
}

function Placeholder({ children }: { children: string }) {
  return <div className="flex h-full items-center justify-center px-4 text-center text-xs text-ink-faint">{children}</div>
}

function PatchView({ lines }: { lines: PatchLine[] }) {
  return <div className="w-max min-w-full py-2 font-mono text-xs leading-5">
    {lines.map((line, index) => <div key={index} className={cn("flex", LINE_CLASS[line.kind])}>
      <span className="w-12 shrink-0 select-none pr-2 text-right tabular-nums text-ink-faint">{line.oldNumber}</span>
      <span className="w-12 shrink-0 select-none pr-2 text-right tabular-nums text-ink-faint">{line.newNumber}</span>
      <span className="w-4 shrink-0 select-none text-center text-ink-faint">{line.kind === "added" ? "+" : line.kind === "removed" ? "-" : ""}</span>
      <span className="whitespace-pre pr-4">{line.text}</span>
    </div>)}
  </div>
}

/** Working-tree changes of one mounted git directory: the changed files beside the patch of the selected one. */
export function DiffViewer({ root }: { root: string }) {
  const [changes, setChanges] = useState<{ root: string; version: number; result: RootDiffFiles | null } | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [patches, setPatches] = useState<{ key: string; files: Record<string, FilePatch> }>({ key: "", files: {} })
  const scrollContainer = useRef<HTMLDivElement>(null)
  const sections = useRef(new Map<string, HTMLElement>())
  const neovim = useNeovimSetting()
  const [editing, setEditing] = useState<NeovimDocument | null>(null)
  const [openingPath, setOpeningPath] = useState<string | null>(null)
  const opening = openingPath !== null
  const [editError, setEditError] = useState("")
  const [revision, setRevision] = useState(0)
  const session = useNeovimTerminal(editing, async (text) => {
    if (!editing) throw new Error("No file selected")
    await writeRootFile(editing.path, text)
    setEditing((current) => current?.key === editing.key ? { ...current, content: text } : current)
    return text
  }, () => {
    setEditing(null)
    setRevision((value) => value + 1)
  })

  useEffect(() => {
    let version = 0
    const load = () => void fetchRootDiffFiles(root).then(
      (result) => setChanges({ root, version: ++version, result }),
      () => setChanges({ root, version: ++version, result: null }),
    )
    load()
    window.addEventListener("focus", load)
    return () => window.removeEventListener("focus", load)
  }, [root, revision])

  const current = changes?.root === root ? changes : null
  const files = current?.result?.files ?? []
  const activePath = editing ? editing.path.slice(root.length + 1) : files.some((file) => file.path === selected) ? selected : files[0]?.path ?? null
  const patchKey = current ? `${root}\0${revision}\0${current.version}` : ""

  useEffect(() => {
    if (!current?.result) return
    let cancelled = false
    let index = 0
    const queue = current.result.files
    const load = async () => {
      while (!cancelled && index < queue.length) {
        const file = queue[index++]
        let patch: FilePatch
        try {
          const diff = await fetchRootFileDiff(root, file.path)
          patch = { lines: diff.supported && diff.patch ? parsePatch(diff.patch) : null, failed: false }
        } catch {
          patch = { lines: null, failed: true }
        }
        if (cancelled) return
        setPatches((previous) => ({ key: patchKey, files: { ...(previous.key === patchKey ? previous.files : {}), [file.path]: patch } }))
      }
    }
    for (let worker = 0; worker < Math.min(4, queue.length); worker++) void load()
    return () => { cancelled = true }
  }, [current, patchKey, root])

  const openEditor = async (filePath: string) => {
    setOpeningPath(filePath)
    setEditError("")
    try {
      const contents = await fetchRootDiffContent(root, filePath)
      const path = `${root}/${filePath}`
      setEditing({ key: path, path, filename: baseName(path), ...contents })
    } catch (error) {
      setEditError(error instanceof Error ? error.message : "Cannot edit this file")
    } finally {
      setOpeningPath(null)
    }
  }
  const jumpToFile = useCallback((path: string) => {
    if (!editing && !opening) {
      setSelected(path)
      setEditError("")
      const container = scrollContainer.current
      const section = sections.current.get(path)
      if (container && section) {
        container.scrollTop += section.getBoundingClientRect().top - container.getBoundingClientRect().top
      }
    }
  }, [editing, opening])
  const trackScroll = () => {
    if (editing || opening) return
    const container = scrollContainer.current
    if (!container) return
    const top = container.getBoundingClientRect().top
    let visible = files[0]?.path ?? null
    for (const file of files) {
      const section = sections.current.get(file.path)
      if (!section || section.getBoundingClientRect().top > top + 8) break
      visible = file.path
    }
    setSelected(visible)
  }

  const loadedPatches = patches.key === patchKey ? patches.files : {}
  const content = !current ? <Placeholder>Loading changes</Placeholder>
    : !current.result ? <Placeholder>Could not read this directory</Placeholder>
    : !current.result.is_git_repository ? <Placeholder>This file mount is not a git repository</Placeholder>
    : !activePath ? <Placeholder>No changes</Placeholder>
    : files.map((file) => {
      const patch = loadedPatches[file.path]
      return <section key={file.path} aria-label={`Diff for ${file.path}`} ref={(element) => {
        if (element) sections.current.set(file.path, element)
        else sections.current.delete(file.path)
      }} className="min-h-24 border-b border-divider/50 last:min-h-full">
        <h3 className="sticky top-0 z-10 flex items-center gap-2 bg-paper px-3 py-2 font-mono text-xs text-ink-muted">
          <span className="min-w-0 flex-1 truncate">{file.path}</span>
          {patch?.lines && <span aria-label={`Line changes in ${file.path}`} className="flex shrink-0 gap-2 tabular-nums">
            <LineCountLabel counts={countLines(patch.lines)} />
          </span>}
          {neovim.enabled && file.status !== "deleted" && <button
            disabled={opening}
            onClick={() => void openEditor(file.path)}
            className="-my-1 shrink-0 rounded px-2 py-1 font-sans text-xs text-ink-faint hover:text-ink disabled:opacity-50"
          >{openingPath === file.path ? "Opening…" : "Edit diff"}</button>}
        </h3>
        {patch?.lines ? <PatchView lines={patch.lines} /> : <div className="px-3 py-6 text-xs text-ink-faint">
          {!patch ? "Loading diff" : patch.failed ? "Could not load diff" : "No text diff for this file"}
        </div>}
      </section>
    })

  return <div className="flex h-full">
    <nav aria-label="Changed files" className="mb-3 ml-3 flex w-72 shrink-0 flex-col rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]">
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        {files.length > 0 && <ChangeSummary root={root} files={files.length} patches={Object.values(loadedPatches)} />}
        <ChangedFiles files={files} patches={loadedPatches} activePath={activePath} onSelect={jumpToFile} root={root} />
      </div>
    </nav>
    <div className="flex min-w-0 flex-1 flex-col">
      <div className="flex h-7 shrink-0 items-center px-3 text-[10px] text-ink-faint">
        <span className="min-w-0 flex-1 truncate">{activePath ?? ""}</span>
        {editing && <span>HEAD · working file · :w saves · :qa returns</span>}
      </div>
      {editError && <p role="alert" className="px-3 py-1 text-xs text-danger">{editError}</p>}
      {editing && <div className="min-h-0 flex-1 overflow-hidden"><NeovimEditor terminal={session.terminal} status={session.status} onRestart={session.restart} className="h-full" /></div>}
      <div ref={scrollContainer} onScroll={trackScroll} aria-label="Diff" className={cn("min-h-0 flex-1 overflow-auto", editing && "hidden")}>
        {content}
      </div>
    </div>
  </div>
}
