"use client"

import { Fragment, memo, useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react"
import { createPortal } from "react-dom"
import { MessageSquare, Pencil, X } from "lucide-react"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { useFileTree } from "@/components/ui/file-tree/useFileTree"
import { fetchRootDiffFiles, fetchRootFileDiff, fetchRootDiffContent, writeRootFile, type DiffStatus, type RootDiffFiles } from "@/lib/backend/api"
import { parsePatch, toSplitRows, type PatchLine } from "@/lib/diff"
import { baseName } from "@/lib/paths"
import { cn } from "@/lib/utils"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, useNeovimTerminal, type NeovimDocument } from "@/features/editor/NeovimEditor"
import { anchorAt, formatReview, locateNote, useReviewNotes, type NoteAnchor, type ReviewNote, type ReviewSide } from "./review"

type FilePatch = { lines: PatchLine[] | null; failed: boolean; stamp: string }

/** Review delivery callback and receiver availability. */
export type ReviewTarget = { send: (text: string) => Promise<void>; canSend: boolean }

/** New note when noteId is null, otherwise an existing note. */
type Draft = { anchor: NoteAnchor; noteId: string | null; comment: string }

type ReviewActions = {
  startDraft: (draft: Draft) => void
  saveDraft: (draft: Draft, comment: string) => void
  cancelDraft: () => void
  removeNote: (id: string) => void
}

type PlacedNotes = { byIndex: Map<number, ReviewNote[]>; outdated: ReviewNote[]; draftIndex: number | null }

const LAYOUTS = ["unified", "split", "auto"] as const
type DiffLayout = (typeof LAYOUTS)[number]

const LAYOUT_STORAGE_KEY = "diff-layout"
const SPLIT_MIN_WIDTH = 1100
const POLL_INTERVAL_MS = 2000
/** Leave room above the destination for the sticky header and three code lines. */
const NOTE_JUMP_OFFSET = 96
const NO_NOTES: ReviewNote[] = []

const STATUS_BADGE: Record<DiffStatus, string> = { modified: "M", added: "A", deleted: "D", renamed: "R", untracked: "U", conflict: "C" }

const LINE_CLASS: Record<PatchLine["kind"], string> = {
  hunk: "bg-surface text-ink-faint",
  added: "bg-diff-added/10 text-diff-added",
  removed: "bg-diff-removed/10 text-diff-removed",
  context: "text-ink-muted",
  note: "text-ink-faint italic",
}

const SIGN: Partial<Record<PatchLine["kind"], string>> = { added: "+", removed: "-" }

const NUMBER_CLASS = "w-12 shrink-0 select-none pr-2 text-right tabular-nums text-ink-faint"
const ICON_BUTTON = "shrink-0 rounded p-1 text-ink-faint hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
const FOCUS_RING = "focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** Nest changed files under their directories, folder ids are `dir:<path>`. */
function buildTree(files: RootDiffFiles["files"], activePath: string | null, select: (path: string) => void, notesByPath: Map<string, ReviewNote[]>): FileTreeItem<string>[] {
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
    const counts = fileCounts(file)
    const noteCount = notesByPath.get(file.path)?.length ?? 0
    siblings.push({
      id: file.path, label: parts[parts.length - 1], type: "element", badge: STATUS_BADGE[file.status],
      meta: (counts || noteCount > 0) && <>
        {noteCount > 0 && <span aria-label={`${noteCount} notes`} className="flex items-center gap-0.5 text-ink-muted"><MessageSquare className="h-2.5 w-2.5" />{noteCount}</span>}
        {counts && <LineCountLabel counts={counts} />}
      </>,
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

function ChangedFiles({ files, activePath, onSelect, root, notesByPath }: {
  files: RootDiffFiles["files"]; activePath: string | null; onSelect: (path: string) => void; root: string; notesByPath: Map<string, ReviewNote[]>
}) {
  const controller = useFileTree({ defaultExpanded: folderIds(files) })
  return <FileTree label={`Changes in ${baseName(root)}`} items={buildTree(files, activePath, onSelect, notesByPath)} controller={controller} />
}

type LineCounts = { added: number; removed: number }

/** Counts arrive with the file list, null for files without a text diff. */
function fileCounts(file: RootDiffFiles["files"][number]): LineCounts | null {
  return file.added === null || file.removed === null ? null : { added: file.added, removed: file.removed }
}

function LineCountLabel({ counts }: { counts: LineCounts }) {
  return <>
    <span className="text-diff-added">+{counts.added}</span>
    <span className="text-diff-removed">−{counts.removed}</span>
  </>
}

/** Pull-request style totals: changed files, then added and removed lines. */
function ChangeSummary({ files }: { files: RootDiffFiles["files"] }) {
  const totals = files.reduce((sum, file) => ({ added: sum.added + (file.added ?? 0), removed: sum.removed + (file.removed ?? 0) }), { added: 0, removed: 0 })
  return <p aria-label="Change summary" className="flex items-center gap-2 px-2 pb-2 text-xs tabular-nums text-ink-faint">
    <span>{files.length} {files.length === 1 ? "file" : "files"}</span>
    <LineCountLabel counts={totals} />
  </p>
}

function Placeholder({ children }: { children: string }) {
  return <div className="flex h-full items-center justify-center px-4 text-center text-xs text-ink-faint">{children}</div>
}

/** Distinguish old/new line numbers in accessible labels. */
function lineName(path: string, side: ReviewSide, line: number | null): string {
  return `${path} ${side === "old" ? "old line" : "line"} ${line}`
}

function LineNumbers({ numbers }: { numbers: (number | null)[] }) {
  return <>{numbers.map((number, column) => <span key={column} className={NUMBER_CLASS}>{number}</span>)}</>
}

function NoteEditor({ draft, actions }: { draft: Draft; actions: ReviewActions }) {
  const [comment, setComment] = useState(draft.comment)
  const selectedText = draft.anchor.selectedText
  return <div className="rounded-lg border border-divider bg-surface px-3 py-2 font-sans focus-within:border-ink-muted">
    {selectedText && <pre className="mb-2 max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-ink-muted">{selectedText}</pre>}
    <textarea
      autoFocus
      aria-label={`Note on ${lineName(draft.anchor.path, draft.anchor.side, draft.anchor.line)}`}
      rows={2}
      value={comment}
      onChange={(event) => setComment(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault()
          actions.cancelDraft()
        } else if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
          event.preventDefault()
          actions.saveDraft(draft, comment)
        }
      }}
      className="block w-full resize-none bg-transparent text-xs leading-5 text-ink outline-none [field-sizing:content]"
    />
  </div>
}

function NoteCard({ note, outdated, actions }: { note: ReviewNote; outdated: boolean; actions: ReviewActions }) {
  return <div className="group rounded-lg border border-divider/60 bg-surface px-3 py-2 font-sans text-xs text-ink">
    <div className="flex items-start gap-2">
      <div className="min-w-0 flex-1">
        {outdated && <p className="mb-1 text-[10px] text-ink-faint">Line {note.line} changed since this note</p>}
        {note.selectedText && <pre className="mb-1.5 max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-ink-muted">{note.selectedText}</pre>}
        <p className="whitespace-pre-wrap break-words leading-5">{note.comment}</p>
        {outdated && <pre className="mt-1.5 overflow-x-auto font-mono text-[11px] leading-4 text-ink-faint">{note.quote.join("\n")}</pre>}
      </div>
      <div className="-my-0.5 flex shrink-0 gap-0.5 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100">
        <button type="button" aria-label="Edit note" onClick={() => actions.startDraft({ anchor: note, noteId: note.id, comment: note.comment })} className={ICON_BUTTON}>
          <Pencil className="h-3 w-3" />
        </button>
        <button type="button" aria-label="Delete note" onClick={() => actions.removeNote(note.id)} className={ICON_BUTTON}>
          <X className="h-3 w-3" />
        </button>
      </div>
    </div>
  </div>
}

/** Notes after one line, an open editor replaces the note it edits. */
function Annotations({ notes, draft, showNewDraft, outdated, actions, className }: {
  notes: ReviewNote[]; draft: Draft | null; showNewDraft: boolean; outdated: boolean; actions: ReviewActions; className: string
}) {
  if (notes.length === 0 && !showNewDraft) return null
  return <div className={cn("space-y-1.5 py-1.5 pr-4", className)}>
    {notes.map((note) => <div key={note.id} data-review-note>
      {draft?.noteId === note.id ? <NoteEditor draft={draft} actions={actions} /> : <NoteCard note={note} outdated={outdated} actions={actions} />}
    </div>)}
    {showNewDraft && draft && <NoteEditor draft={draft} actions={actions} />}
  </div>
}

function placeNotes(notes: ReviewNote[], draft: Draft | null, lines: PatchLine[]): PlacedNotes {
  const byIndex = new Map<number, ReviewNote[]>()
  const outdated: ReviewNote[] = []
  for (const note of notes) {
    const index = locateNote(note, lines)
    if (index === null) outdated.push(note)
    else byIndex.set(index, [...(byIndex.get(index) ?? []), note])
  }
  const draftIndex = draft && draft.noteId === null ? locateNote(draft.anchor, lines) : null
  return { byIndex, outdated, draftIndex }
}

type PatchViewProps = { path: string; lines: PatchLine[]; split: boolean; placed: PlacedNotes; draft: Draft | null; actions: ReviewActions }

/** Memoized: a patch keeps its `lines` array until its file changes, so polls and scrolling skip the large line lists. */
const PatchView = memo(function PatchView({ path, lines, split, placed, draft, actions }: PatchViewProps) {
  const rows = useMemo(() => split ? toSplitRows(lines) : null, [split, lines])
  const annotations = (index: number | null, className: string) => index !== null && <Annotations
    notes={placed.byIndex.get(index) ?? NO_NOTES} draft={draft} showNewDraft={placed.draftIndex === index}
    outdated={false} actions={actions} className={className}
  />

  if (rows) {
    const cell = (index: number | null, side: ReviewSide) => {
      const line = index === null ? null : lines[index]!
      if (!line) return <div className="bg-surface/40" />
      const number = side === "old" ? line.oldNumber : line.newNumber
      return <div className={cn("flex min-w-0", LINE_CLASS[line.kind])}>
        <LineNumbers numbers={[number]} />
        <span className="w-4 shrink-0 select-none text-center text-ink-faint">{SIGN[line.kind]}</span>
        <span data-diff-code data-path={path} data-index={index} data-side={side} data-column={side} className="min-w-0 whitespace-pre-wrap break-all pr-4">{line.text}</span>
      </div>
    }
    return <div className="py-2 font-mono text-xs leading-5">
      {rows.map((row, position) => "full" in row
        ? <div key={position} className={cn("pl-16", LINE_CLASS[lines[row.full]!.kind])}>{lines[row.full]!.text}</div>
        : <Fragment key={position}>
          <div className="grid grid-cols-2">{cell(row.old, "old")}{cell(row.new, "new")}</div>
          {annotations(row.old, "pl-16")}
          {row.new !== row.old && annotations(row.new, "pl-16")}
        </Fragment>)}
    </div>
  }

  return <div className="w-max min-w-full py-2 font-mono text-xs leading-5">
    {lines.map((line, index) => {
      const side: ReviewSide = line.kind === "removed" ? "old" : "new"
      return <Fragment key={index}>
        <div className={cn("flex", LINE_CLASS[line.kind])}>
          <LineNumbers numbers={[line.oldNumber, line.newNumber]} />
          <span className="w-4 shrink-0 select-none text-center text-ink-faint">{SIGN[line.kind]}</span>
          <span data-diff-code data-path={path} data-index={index} data-side={side} data-column="unified" className="whitespace-pre pr-4">{line.text}</span>
        </div>
        {annotations(index, "sticky left-0 w-[min(44rem,100cqw)] pl-28")}
      </Fragment>
    })}
  </div>
})

/** One file of the diff, props are primitives or stable references so unchanged files skip re-rendering. */
const FileSection = memo(function FileSection({ path, added, removed, patch, editable, opening, openingThis, onEdit, register, split, notes, draft, actions }: {
  path: string; added: number | null; removed: number | null; patch: FilePatch | undefined
  editable: boolean; opening: boolean; openingThis: boolean
  onEdit: (path: string) => void; register: (path: string, element: HTMLElement | null) => void
  split: boolean; notes: ReviewNote[]; draft: Draft | null; actions: ReviewActions
}) {
  const lines = patch?.lines ?? null
  const placed = useMemo(() => placeNotes(notes, draft, lines ?? []), [notes, draft, lines])
  return <section aria-label={`Diff for ${path}`} ref={(element) => register(path, element)} className="min-h-24 border-b border-divider/50 last:min-h-full">
    <h3 className="sticky left-0 top-0 z-10 flex items-center gap-2 bg-paper px-3 py-2 font-mono text-xs text-ink-muted">
      <span className="min-w-0 flex-1 truncate">{path}</span>
      {added !== null && removed !== null && <span aria-label={`Line changes in ${path}`} className="flex shrink-0 gap-2 tabular-nums">
        <LineCountLabel counts={{ added, removed }} />
      </span>}
      {editable && <button
        disabled={opening}
        onClick={() => onEdit(path)}
        className="-my-1 shrink-0 rounded px-2 py-1 font-sans text-xs text-ink-faint hover:text-ink disabled:opacity-50"
      >{openingThis ? "Opening…" : "Edit diff"}</button>}
    </h3>
    {patch && <Annotations
      notes={placed.outdated} draft={draft} showNewDraft={false} outdated actions={actions}
      className="sticky left-0 w-[min(44rem,100cqw)] pl-3"
    />}
    {lines && placed ? <PatchView path={path} lines={lines} split={split} placed={placed} draft={draft} actions={actions} /> : <div className="px-3 py-6 text-xs text-ink-faint">
      {!patch ? "Loading diff" : patch.failed ? "Could not load diff" : "No text diff for this file"}
    </div>}
  </section>
})

function storedLayout(): DiffLayout {
  try {
    return LAYOUTS.find((layout) => layout === localStorage.getItem(LAYOUT_STORAGE_KEY)) ?? "auto"
  } catch {
    return "auto"
  }
}

/** Persist layout choice and resolve auto from pane width. */
function useDiffLayout(container: RefObject<HTMLElement | null>) {
  const [layout, setLayout] = useState(storedLayout)
  const [wide, setWide] = useState(false)
  useEffect(() => {
    const element = container.current
    if (!element || typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(([entry]) => setWide(entry!.contentRect.width >= SPLIT_MIN_WIDTH))
    observer.observe(element)
    return () => observer.disconnect()
  }, [container])
  const choose = (next: DiffLayout) => {
    setLayout(next)
    localStorage.setItem(LAYOUT_STORAGE_KEY, next)
  }
  return { layout, choose, split: layout === "split" || (layout === "auto" && wide) }
}

/** Scroll `container` to the next or previous review note below its sticky file header. */
function jumpToNote(container: HTMLElement, direction: 1 | -1) {
  const anchor = container.getBoundingClientRect().top + NOTE_JUMP_OFFSET
  const offsets = [...container.querySelectorAll("[data-review-note]")].map((note) => note.getBoundingClientRect().top - anchor)
  const target = direction > 0 ? offsets.find((offset) => offset > 1) : offsets.reverse().find((offset) => offset < -1)
  if (target !== undefined) container.scrollTop += target
}

/** Capture code only, excluding gutters and comments, within one file and split-view side. */
function selectedComment(container: HTMLElement, patches: Record<string, FilePatch>) {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount !== 1) return null
  const range = selection.getRangeAt(0)
  const startElement = range.startContainer.nodeType === Node.ELEMENT_NODE ? range.startContainer as Element : range.startContainer.parentElement
  const endElement = range.endContainer.nodeType === Node.ELEMENT_NODE ? range.endContainer as Element : range.endContainer.parentElement
  const start = startElement?.closest<HTMLElement>("[data-diff-code]")
  const end = endElement?.closest<HTMLElement>("[data-diff-code]")
  if (!start || !end || !container.contains(start) || !container.contains(end) || start.dataset.path !== end.dataset.path || start.dataset.column !== end.dataset.column) return null
  const path = start.dataset.path!
  const lines = patches[path]?.lines
  if (!lines) return null
  const text: string[] = []
  let lastCode: HTMLElement | null = null
  let bounds: DOMRect | null = null
  for (const code of start.closest("section")!.querySelectorAll<HTMLElement>("[data-diff-code]")) {
    if (code.dataset.column !== start.dataset.column || !range.intersectsNode(code)) continue
    const part = document.createRange()
    part.selectNodeContents(code)
    if (range.compareBoundaryPoints(Range.START_TO_START, part) > 0) part.setStart(range.startContainer, range.startOffset)
    if (range.compareBoundaryPoints(Range.END_TO_END, part) < 0) part.setEnd(range.endContainer, range.endOffset)
    const selected = part.toString()
    const blankInside = code.textContent === "" && range.compareBoundaryPoints(Range.START_TO_START, part) < 0 && range.compareBoundaryPoints(Range.END_TO_END, part) > 0
    if (selected || blankInside) {
      text.push(selected)
      lastCode = code
      const rectangle = part.getBoundingClientRect()
      bounds = rectangle.height > 0 ? rectangle : code.getBoundingClientRect()
    }
  }
  const selectedText = text.join("\n")
  if (!selectedText.trim() || !lastCode || !bounds) return null
  const anchor = anchorAt(path, lines, Number(lastCode.dataset.index), lastCode.dataset.side as ReviewSide)
  if (!anchor) return null
  return {
    anchor: { ...anchor, selectedText },
    // Match the bubble SVG's tip at (22, 1) to the highlight's lower-left corner.
    left: bounds.left - 22,
    top: bounds.bottom - 1,
  }
}

/** Show working-tree changes with optional line notes and batched review delivery. */
export function DiffViewer({ root, review }: { root: string; review?: ReviewTarget }) {
  const [changes, setChanges] = useState<{ root: string; result: RootDiffFiles | null } | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [patches, setPatches] = useState<{ root: string; files: Record<string, FilePatch> }>({ root: "", files: {} })
  const patchesRef = useRef(patches)
  useEffect(() => { patchesRef.current = patches }, [patches])
  const scrollContainer = useRef<HTMLDivElement>(null)
  const sections = useRef(new Map<string, HTMLElement>())
  const neovim = useNeovimSetting()
  const [editing, setEditing] = useState<NeovimDocument | null>(null)
  const [openingPath, setOpeningPath] = useState<string | null>(null)
  const opening = openingPath !== null
  const [editError, setEditError] = useState("")
  const [revision, setRevision] = useState(0)
  const { layout, choose: chooseLayout, split } = useDiffLayout(scrollContainer)
  const { notes, update: updateNotes } = useReviewNotes(root)
  const [selectionAction, setSelectionAction] = useState<ReturnType<typeof selectedComment>>(null)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [sending, setSending] = useState(false)
  const [sendError, setSendError] = useState("")
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    const update = () => {
      const container = scrollContainer.current
      setSelectionAction(container && patchesRef.current.root === root ? selectedComment(container, patchesRef.current.files) : null)
    }
    const dismiss = () => setSelectionAction(null)
    document.addEventListener("selectionchange", update)
    window.addEventListener("scroll", dismiss, true)
    window.addEventListener("resize", dismiss)
    return () => {
      document.removeEventListener("selectionchange", update)
      window.removeEventListener("scroll", dismiss, true)
      window.removeEventListener("resize", dismiss)
    }
  }, [root])
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
    let inFlight = false
    const load = () => {
      if (inFlight || document.hidden) return
      inFlight = true
      void fetchRootDiffFiles(root).then(
        (result) => setChanges((previous) => previous?.root === root && JSON.stringify(previous.result) === JSON.stringify(result) ? previous : { root, result }),
        () => setChanges((previous) => previous?.root === root ? previous : { root, result: null }),
      ).finally(() => { inFlight = false })
    }
    load()
    const timer = setInterval(load, POLL_INTERVAL_MS)
    window.addEventListener("focus", load)
    document.addEventListener("visibilitychange", load)
    return () => {
      clearInterval(timer)
      window.removeEventListener("focus", load)
      document.removeEventListener("visibilitychange", load)
    }
  }, [root, revision])

  const current = changes?.root === root ? changes : null
  const files = current?.result?.files ?? []
  const activePath = editing ? editing.path.slice(root.length + 1) : files.some((file) => file.path === selected) ? selected : files[0]?.path ?? null

  useEffect(() => {
    if (!current?.result) return
    let cancelled = false
    const known = patchesRef.current.root === root ? patchesRef.current.files : {}
    const queue = current.result.files.filter((file) => known[file.path]?.stamp !== file.stamp)
    let index = 0
    const load = async () => {
      while (!cancelled && index < queue.length) {
        const file = queue[index++]
        let patch: FilePatch
        try {
          const diff = await fetchRootFileDiff(root, file.path)
          patch = { lines: diff.supported && diff.patch ? parsePatch(diff.patch) : null, failed: false, stamp: file.stamp }
        } catch {
          patch = { lines: null, failed: true, stamp: file.stamp }
        }
        if (cancelled) return
        setPatches((previous) => ({ root, files: { ...(previous.root === root ? previous.files : {}), [file.path]: patch } }))
      }
    }
    for (let worker = 0; worker < Math.min(4, queue.length); worker++) void load()
    return () => { cancelled = true }
  }, [current, root])

  const actions = useMemo<ReviewActions>(() => ({
    startDraft: setDraft,
    cancelDraft: () => setDraft(null),
    saveDraft: (saved, comment) => {
      const text = comment.trim()
      updateNotes((current) => saved.noteId === null
        ? text ? [...current, { ...saved.anchor, id: crypto.randomUUID(), comment: text }] : current
        : text ? current.map((note) => note.id === saved.noteId ? { ...note, comment: text } : note) : current.filter((note) => note.id !== saved.noteId))
      setDraft(null)
    },
    removeNote: (id) => updateNotes((current) => current.filter((note) => note.id !== id)),
  }), [updateNotes])
  const notesByPath = useMemo(() => {
    const grouped = new Map<string, ReviewNote[]>()
    for (const note of notes) grouped.set(note.path, [...(grouped.get(note.path) ?? []), note])
    return grouped
  }, [notes])

  const sendReview = async () => {
    if ((review && !review.canSend) || sending || notes.length === 0) return
    const sent = new Set(notes)
    setSending(true)
    setSendError("")
    setCopied(false)
    try {
      const loaded = patchesRef.current.root === root ? patchesRef.current.files : {}
      const currentPatches = Object.fromEntries(files.map((file) => {
        const patch = loaded[file.path]
        return [file.path, patch?.stamp === file.stamp ? patch.lines : null]
      }))
      const text = formatReview(notes, currentPatches)
      if (review) {
        await review.send(text)
        updateNotes((current) => current.filter((note) => !sent.has(note)))
      } else {
        await navigator.clipboard.writeText(text)
        setCopied(true)
      }
    } catch (error) {
      setSendError(error instanceof Error ? error.message : "Could not export the review")
    } finally {
      setSending(false)
    }
  }

  const openEditor = useCallback(async (filePath: string) => {
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
  }, [root])
  const editFile = useCallback((path: string) => void openEditor(path), [openEditor])
  const registerSection = useCallback((path: string, element: HTMLElement | null) => {
    if (element) sections.current.set(path, element)
    else sections.current.delete(path)
  }, [])
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
  const scrollFrame = useRef(0)
  /** Find the file at the top of the viewport at most once per frame, scroll events fire far more often. */
  const trackScroll = () => {
    if (editing || opening || scrollFrame.current) return
    scrollFrame.current = requestAnimationFrame(() => {
      scrollFrame.current = 0
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
    })
  }
  useEffect(() => () => cancelAnimationFrame(scrollFrame.current), [])

  const loadedPatches = patches.root === root ? patches.files : {}
  const content = !current ? <Placeholder>Loading changes</Placeholder>
    : !current.result ? <Placeholder>Could not read this directory</Placeholder>
    : !current.result.is_git_repository ? <Placeholder>This file mount is not a git repository</Placeholder>
    : !activePath ? <Placeholder>No changes</Placeholder>
    : files.map((file) => <FileSection
      key={file.path} path={file.path} added={file.added} removed={file.removed} patch={loadedPatches[file.path]}
      editable={neovim.enabled && file.status !== "deleted"} opening={opening} openingThis={openingPath === file.path}
      onEdit={editFile} register={registerSection}
      split={split} notes={notesByPath.get(file.path) ?? NO_NOTES} draft={draft?.anchor.path === file.path ? draft : null} actions={actions}
    />)

  return <div className="flex h-full">
    <nav aria-label="Changed files" className="mb-3 ml-3 flex w-72 shrink-0 flex-col rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]">
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        {files.length > 0 && <ChangeSummary files={files} />}
        <ChangedFiles files={files} activePath={activePath} onSelect={jumpToFile} root={root} notesByPath={notesByPath} />
        {current?.result && [...notesByPath].filter(([path]) => !files.some((file) => file.path === path)).map(([path, missingNotes]) => <div key={path} className="mt-3">
          <p className="truncate px-1 text-xs text-ink-faint">{path} · no longer in diff</p>
          <Annotations notes={missingNotes} draft={draft} showNewDraft={false} outdated actions={actions} className="pl-1" />
        </div>)}
      </div>
      {notes.length > 0 && <div aria-label="Review" className="border-t border-divider/50 px-3 py-2.5">
        <div className="flex items-center gap-2 text-xs text-ink-muted">
          <span className="flex-1 tabular-nums">{notes.length} {notes.length === 1 ? "note" : "notes"}</span>
          <button
            type="button"
            disabled={(review && !review.canSend) || sending}
            onClick={() => void sendReview()}
            className={cn("rounded-full bg-ink px-3 py-1.5 text-xs font-medium text-paper transition-opacity hover:opacity-90 focus-visible:outline-offset-2 disabled:opacity-40", FOCUS_RING)}
          >{sending ? "Exporting…" : review ? "Send review" : "Copy review"}</button>
        </div>
        {review && !review.canSend && <p className="mt-1.5 text-[11px] text-ink-faint">Sendable once the agent waits for you</p>}
        {!review && copied && <p role="status" className="mt-1.5 text-[11px] text-ink-faint">Review copied</p>}
        {sendError && <p role="alert" className="mt-1.5 text-xs text-danger">{sendError}</p>}
      </div>}
    </nav>
    <div className="flex min-w-0 flex-1 flex-col">
      <div className="flex h-7 shrink-0 items-center gap-3 px-3 text-[10px] text-ink-faint">
        <span className="min-w-0 flex-1 truncate">{activePath ?? ""}</span>
        {editing ? <span>HEAD · working file · :w saves · :qa returns</span> : <div role="group" aria-label="Diff layout" className="flex shrink-0 gap-0.5">
          {LAYOUTS.map((option) => <button
            key={option}
            type="button"
            aria-pressed={layout === option}
            onClick={() => chooseLayout(option)}
            className={cn("rounded px-1.5 py-0.5 capitalize", FOCUS_RING, layout === option ? "text-ink" : "hover:text-ink")}
          >{option}</button>)}
        </div>}
      </div>
      {editError && <p role="alert" className="px-3 py-1 text-xs text-danger">{editError}</p>}
      {editing && <div className="min-h-0 flex-1 overflow-hidden"><NeovimEditor terminal={session.terminal} status={session.status} onRestart={session.restart} className="h-full" /></div>}
      <div
        ref={scrollContainer}
        role="region"
        aria-label="Diff"
        tabIndex={0}
        onScroll={trackScroll}
        onPointerUp={(event) => setSelectionAction(selectedComment(event.currentTarget, loadedPatches))}
        onKeyUp={(event) => setSelectionAction(selectedComment(event.currentTarget, loadedPatches))}
        onKeyDown={(event) => {
          const arrow = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0
          if (!arrow || notes.length === 0 || event.target instanceof HTMLTextAreaElement || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return
          event.preventDefault()
          jumpToNote(event.currentTarget, arrow)
        }}
        className={cn("@container min-h-0 flex-1 overflow-auto focus-visible:outline focus-visible:outline-1 focus-visible:-outline-offset-1 focus-visible:outline-ink-muted/40", editing && "hidden")}
      >
        {content}
      </div>
      {selectionAction && !editing && createPortal(<button
        type="button"
        aria-label="Add comment to selection"
        onPointerDown={(event) => event.preventDefault()}
        onClick={() => {
          setDraft({ anchor: selectionAction.anchor, noteId: null, comment: "" })
          setSelectionAction(null)
          window.getSelection()?.removeAllRanges()
        }}
        style={{ left: selectionAction.left, top: selectionAction.top }}
        className={cn("fixed z-50 h-8 w-8 rounded text-ink-muted hover:text-ink", FOCUS_RING)}
      >
        <svg aria-hidden="true" viewBox="0 0 32 32" className="h-8 w-8 fill-paper stroke-current" strokeWidth="2.25" strokeLinejoin="round">
          <path d="M22 1v6q0 3 3 3h1q5 0 5 6v8q0 6-6 6H7q-6 0-6-6v-8q0-6 6-6h5q2 0 4-2z" />
        </svg>
      </button>, document.body)}
    </div>
  </div>
}
