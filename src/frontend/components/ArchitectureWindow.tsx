"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { Box, FileType, Layers, LayoutGrid, ListPlus, Network, PanelLeftClose, PanelLeftOpen, Plus, Trash2 } from "lucide-react"
import { applyArchitectureOperations, deleteArchitectureSources, readArchitecture } from "@/lib/api"
import {
  architectureSlug, architectureSource, diffGraph, fitGraph, graphFromView, labelFontsReady, nodeMinWidth,
  type ArchitectureElement, type ArchitectureOperation, type ArchitectureTreeEntry, type ArchitectureViewNode, type ArchitectureWorkspace, type DiagramGraph, type DiagramNode, type EdgeStyle,
} from "@/lib/architecture"
import { getDeviceId } from "@/lib/deviceId"
import { subscribeToChanges } from "@/lib/syncStream"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { cn } from "@/lib/utils"
import { ArchitectureDiagramSurface } from "./ArchitectureDiagramSurface"
import { DocumentEditor } from "./DocumentEditor"

type Mutate = (write: () => Promise<ArchitectureWorkspace>) => Promise<ArchitectureWorkspace>

/** The server's reason for a failed load or write, or `fallback` when it gave none. */
const failure = (cause: unknown, fallback: string) => cause instanceof Error && cause.message ? cause.message : fallback

/**
 * A `.c4` file of a project's architecture, drawn through the view it opens
 * in: a system's own `view x of x`, its system's view for a module, or a
 * saved view. Reads and writes of the model run one at a time in order; each
 * write is validated by the server against the model as it is then, so a
 * stale edit is refused with a reason instead of overwriting newer work.
 */
function useArchitectureView(slug: string, file: string) {
  const [workspace, setWorkspace] = useState<ArchitectureWorkspace | null>(null)
  // A refused edit, cleared by the next one; and a failed read, cleared by the next successful read.
  const [error, setError] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<DiagramNode[]>([])
  const [optimistic, setOptimistic] = useState<DiagramGraph | null>(null)
  const queueRef = useRef<Promise<void>>(Promise.resolve())
  const inFlightRef = useRef(0)

  // Queues a write or a read; its answer, the whole workspace, replaces the shown one.
  // A failure rejects with the server's reason, which the caller shows.
  const mutate: Mutate = useCallback((write) => {
    inFlightRef.current += 1
    const result = queueRef.current.then(async () => {
      try {
        const next = await write()
        setWorkspace(next)
        return next
      } finally {
        inFlightRef.current -= 1
        // Only once the queue drains: an earlier answer must not wipe a later edit's preview.
        if (inFlightRef.current === 0) setOptimistic(null)
      }
    })
    queueRef.current = result.then(() => undefined, () => undefined)
    return result
  }, [])

  // The only way the model is read: on opening, when another device changes it, and after a text save.
  // A read waits its turn behind queued writes, so it never replaces a newer write's answer. While it
  // waits, `queuedReadRef` holds its slug, and further requests for that slug share it.
  const queuedReadRef = useRef<string | null>(null)
  const refresh = useCallback(() => {
    if (queuedReadRef.current === slug) return
    queuedReadRef.current = slug
    mutate(() => {
      queuedReadRef.current = null
      return readArchitecture(slug)
    }).then(() => setLoadError(null), (cause: unknown) => setLoadError(failure(cause, "Could not load the architecture")))
  }, [mutate, slug])

  useEffect(() => {
    setWorkspace(null)
    setError(null)
    setLoadError(null)
    refresh()
  }, [refresh])

  // Writes from this device are already reflected locally; anyone else's reload the model.
  useEffect(() => subscribeToChanges((event) => {
    if (event.resource_kind !== "document" || !event.resource_key.startsWith(`graph/${slug}/`)) return
    if (event.device_id !== getDeviceId()) refresh()
  }), [slug, refresh])

  // How the tree lists `file`, and the view it opens in (a module opens its system's).
  const entry = useMemo(() => workspace?.model.tree.find((candidate) => candidate.path === file) ?? null, [workspace, file])
  const view = useMemo(() => workspace?.model.views.find((candidate) => candidate.id === entry?.view) ?? null, [workspace, entry])
  // Loaded, but `file` no longer exists or opens no view.
  const missing = !workspace || view ? null : entry ? "view" : "file"

  useEffect(() => { setDrafts([]) }, [view?.id])

  const serverGraph = useMemo(() => view ? graphFromView(view) : null, [view])
  // Sized and spaced to fit their labels once, before drawing, and again once the real font has loaded.
  const [fontsLoaded, setFontsLoaded] = useState(false)
  useEffect(() => { void labelFontsReady.then(() => setFontsLoaded(true)) }, [])
  // What the surface draws: the server's view, untitled drafts, and any edit still in flight.
  const graph = useMemo(() => {
    const raw = optimistic ?? (serverGraph && { ...serverGraph, nodes: [...serverGraph.nodes, ...drafts] })
    return raw && { ...raw, nodes: fitGraph(raw.nodes, (node) => nodeMinWidth(node.title, node.kind)) }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fontsLoaded invalidates the measurements
  }, [optimistic, serverGraph, drafts, fontsLoaded])
  const graphRef = useRef(graph)
  graphRef.current = graph

  // A write from the diagram, whose refusal the window shows.
  const enqueue = useCallback((write: () => Promise<ArchitectureWorkspace>) => {
    mutate(write).catch((cause: unknown) => setError(failure(cause, "The architecture could not be changed")))
  }, [mutate])

  // Runs operations the surface cannot express, e.g. adding to or removing from the view.
  const run = useCallback((ops: ArchitectureOperation[]) => {
    if (ops.length === 0) return
    setError(null)
    enqueue(() => applyArchitectureOperations(slug, ops, file))
  }, [enqueue, slug, file])

  const onChange = useCallback((next: DiagramGraph) => {
    const before = graphRef.current
    if (!before || !view || !workspace) return
    // A node drawn outside every element belongs to the view's scope (`view x of x`); the server knows that.
    const change = diffGraph(before, next, view.id, workspace.model.kinds)
    if (change.rejected) { setError(change.rejected); return }
    setDrafts(change.drafts)
    if (change.ops.length === 0) return
    setError(null)
    setOptimistic(next)
    enqueue(() => applyArchitectureOperations(slug, change.ops, file))
  }, [enqueue, slug, file, view, workspace])

  // Save LikeC4's layout the first time an editable view is drawn, so later
  // model edits (which re-run that layout) never move what the user has already seen.
  const pinnedRef = useRef(new Set<string>())
  useEffect(() => {
    if (!view?.editable || view.manual || pinnedRef.current.has(view.id)) return
    pinnedRef.current.add(view.id)
    run([{ op: "layout", view: view.id, nodes: {}, edges: {} }])
  }, [run, view])

  return { workspace, entry, view, graph, missing, error: error ?? loadError, onChange, run, mutate, refresh }
}

// An architecture window owns only its canvas-local frame, connection style, and
// the file it is on; the model lives in the project's LikeC4 workspace (every
// `.c4` file). The tree on its left moves between systems, modules, and saved
// views; the diagram is the view the file opens in (a module opens its system's,
// with the module selected), or the file itself as text. It is rendered at real
// layout size: CSS scaling would desynchronise React Flow's handles from the pointer.
export function ArchitectureWindow({ path, maximized, hostScale, edgeStyle = "elbow", onEdgeStyleChange, onNavigate }: {
  path: string
  maximized: boolean
  hostScale: number
  /** How connections are drawn: a window preference, LikeC4 has no such setting. */
  edgeStyle?: EdgeStyle
  onEdgeStyleChange: (style: EdgeStyle) => void
  /** Points the window at another `graph/<slug>/<file>`; an empty file means "the first system". */
  onNavigate: (path: string) => void
}) {
  const slug = architectureSlug(path)
  const file = architectureSource(path)
  const [treeOpen, setTreeOpen] = useState(true)
  const [mode, setMode] = useState<"diagram" | "text">("diagram")
  const { workspace, entry, view, graph, missing, error, onChange, run, mutate, refresh } = useArchitectureView(slug, file)
  const open = useCallback((next: string) => onNavigate(`graph/${slug}/${next}`), [onNavigate, slug])
  // A new window, or one whose file was just deleted, opens the first system.
  useEffect(() => {
    if (file || !workspace) return
    const first = workspace.model.tree.find((candidate) => candidate.role === "system") ?? workspace.model.tree.find((candidate) => candidate.view)
    if (first) open(first.path)
  }, [file, workspace, open])
  // Removing from a saved view leaves the element in the model; the user is then asked whether to delete it there too.
  const [offered, setOffered] = useState<string[]>([])
  const removeFromView = useCallback((elements: string[]) => {
    if (!view || elements.length === 0) return
    run([{ op: "removeFromView", view: view.id, elements }])
    setOffered(elements)
  }, [run, view])
  const offeredElements = workspace ? workspace.model.elements.filter((element) => offered.includes(element.id)) : []
  const answer = useCallback((remove: boolean) => {
    if (remove) run([{ op: "delete", elements: offered, relations: [] }])
    setOffered([])
  }, [run, offered])
  // The dialog has the keyboard: Enter deletes, Escape keeps, whatever holds focus.
  useEffect(() => {
    if (offered.length === 0) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Enter" && event.key !== "Escape") return
      event.preventDefault()
      event.stopPropagation()
      answer(event.key === "Enter")
    }
    window.addEventListener("keydown", onKey, true)
    return () => window.removeEventListener("keydown", onKey, true)
  }, [offered, answer])
  // Another view or file is another question.
  useEffect(() => { setOffered([]) }, [view?.id])
  // LikeC4's own complaints about the source, e.g. an unresolved reference; it counts lines from zero.
  const diagnostics = workspace?.model.errors ?? []
  const notice = (text: string, action?: ReactNode) => (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 px-4 text-center text-[10px] text-ink-faint">
      <p>{text}</p>
      {action}
    </div>
  )
  let body: ReactNode
  if (!workspace) body = notice(error ? "" : "Loading architecture…")
  else if (!file) body = notice("Start with a system: its modules are the parts inside it.")
  else if (missing === "file") body = notice(`${file} no longer exists.`)
  else if (mode === "text") body = (
    <div className="relative min-h-0 flex-1">
      {/* Saved text is this device's own write, which the change feed skips, so reread explicitly. */}
      <DocumentEditor key={file} path={`graph/${slug}/${file}`} slug={slug} overlay onClose={() => setMode("diagram")} onSaved={refresh} />
    </div>
  )
  else if (missing === "view") body = notice(`${file} declares no view yet.`, (
    <button type="button" className="architecture-diagram-button" onClick={() => run([{ op: "addFileView" }])}>Add a view of this file</button>
  ))
  else if (!view || !graph) body = notice("Loading architecture…")
  else {
    // A landscape view shows what it names; a scoped one shows its system's surroundings.
    const landscape = view.editable && view.scope === null
    const toolbar = !view.editable
      ? <span className="text-[10px] text-ink-faint">Read-only: this view uses filters or styles, edit it as text</span>
      : <>
        {landscape && <>
          <span className="architecture-diagram-toolbar-delimiter" aria-hidden="true">|</span>
          <ArchitectureViewPicker elements={workspace.model.elements} shown={view.nodes} onInclude={(elements, descendants) => run([{ op: "includeInView", view: view.id, elements, descendants }])} />
        </>}
        {/* Hands every position back to LikeC4: arrows decide the ranks, the rest lines up. */}
        {view.manual && (
          <button type="button" className="architecture-diagram-toolbar-symbol" style={{ width: "auto", padding: "0 6px", fontSize: 11 }}
            onClick={() => run([{ op: "layout", view: view.id, nodes: Object.fromEntries(view.nodes.map((node) => [node.id, null])), edges: Object.fromEntries(view.edges.map((edge) => [edge.id, null])) }])}>
            <LayoutGrid size={13} /> Auto layout
          </button>
        )}
      </>
    body = (
      <ArchitectureDiagramSurface
        key={view.id} graph={graph} onChange={onChange} readOnly={!view.editable} className="min-h-0 flex-1" autoFit={maximized} hostScale={hostScale}
        edgeStyle={edgeStyle} onEdgeStyleChange={onEdgeStyleChange}
        toolbar={toolbar} onRemoveFromView={landscape ? removeFromView : undefined}
        select={entry?.role === "module" ? entry.element : null} kinds={workspace.model.kinds} topLevelDrawing={!landscape}
      />
    )
  }
  return (
    <div className="flex h-full">
      {treeOpen && workspace && <ArchitectureTree slug={slug} workspace={workspace} current={file} onOpen={open} onCollapse={() => setTreeOpen(false)} mutate={mutate} />}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-6 shrink-0 items-center gap-2 border-b border-divider/50 px-2 text-[10px] text-ink-faint">
          {!treeOpen && <button type="button" aria-label="Show files" onClick={() => setTreeOpen(true)} className="rounded p-0.5 hover:text-ink"><PanelLeftOpen className="h-3 w-3" /></button>}
          <span className="min-w-0 flex-1 truncate">{file}</span>
          {entry && (
            <div role="group" aria-label="Show as" className="flex gap-px rounded bg-surface p-px">
              {(["diagram", "text"] as const).map((option) => (
                <button key={option} type="button" aria-pressed={mode === option} onClick={() => setMode(option)} className={cn("rounded px-1.5 py-px transition-colors", mode === option ? "bg-surface-elevated text-ink" : "hover:text-ink")}>
                  {option === "diagram" ? "Diagram" : "Text"}
                </button>
              ))}
            </div>
          )}
        </div>
        {(error || diagnostics.length > 0) && (
          <div className="max-h-20 shrink-0 overflow-y-auto px-2 py-1 text-[10px] text-danger">
            {error && <p role="alert">{error}</p>}
            {diagnostics.map((diagnostic, index) => <p key={index}>{diagnostic.file}:{diagnostic.line + 1}: {diagnostic.message}</p>)}
          </div>
        )}
        {offeredElements.length > 0 && createPortal(
          // Portaled: the window sits inside the canvas's transform, which would confine a fixed overlay.
          <div className="fixed inset-0 z-[100] flex items-center justify-center bg-backdrop backdrop-blur-[2px]" onClick={() => answer(false)}>
            <div role="alertdialog" aria-label="Remove reference" onClick={(event) => event.stopPropagation()}
              className="mx-4 flex flex-col gap-3 rounded-xl border border-divider bg-paper px-5 py-4 text-sm text-ink shadow-[var(--shadow-xl)]">
              <span className="flex items-center gap-2">
                <Trash2 className="h-4 w-4" />
                Remove reference from {[...new Set(offeredElements.map((element) => element.file ?? "the model"))].join(", ")}?
              </span>
              <span className="flex justify-end gap-2 text-xs">
                <button type="button" onClick={() => answer(true)} className="rounded-lg border border-divider bg-surface px-3 py-1.5 text-ink hover:border-ink-muted">Yes (Enter)</button>
                <button type="button" onClick={() => answer(false)} className="rounded-lg border border-divider px-3 py-1.5 text-ink-muted hover:border-ink-muted hover:text-ink">No (Esc)</button>
              </span>
            </div>
          </div>,
          document.body,
        )}
        {body}
      </div>
    </div>
  )
}

// The project's architecture as files: systems (a folder whose own file declares
// one system) with their modules, saved views, and anything else hand-written.
// Creating writes LikeC4 from templates; deleting a system takes its modules
// along, and the server drops the connections and view entries naming them.
// A deleted row leaves the tree at once, but its files (and so the diagram) stay
// until the delete commits when the undo window closes.
function ArchitectureTree({ slug, workspace, current, onOpen, onCollapse, mutate }: {
  slug: string
  workspace: ArchitectureWorkspace
  current: string
  onOpen: (file: string) => void
  onCollapse: () => void
  mutate: Mutate
}) {
  const [naming, setNaming] = useState<{ kind: "system" | "view" } | { kind: "module", system: string } | null>(null)
  const { schedule, pending } = useUndoDelete()
  const [problem, setProblem] = useState<string | null>(null)
  // Read when a delete commits, which may be after the user has moved to another file.
  const currentRef = useRef(current)
  currentRef.current = current
  const { tree, elements, views } = workspace.model
  // One undo key per delete, named after its first path: a system's key hides its modules with it.
  const deleteKey = (path: string) => `architecture:${slug}:${path}`
  const allSystems = tree.filter((entry) => entry.role === "system")
  const systems = allSystems.filter((entry) => !pending(deleteKey(entry.path)))
  const systemIds = new Set(allSystems.map((entry) => entry.element))
  const systemOf = (entry: ArchitectureTreeEntry) => entry.element?.split(".")[0] ?? ""
  const modulesOf = (system: string) => tree.filter((entry) => entry.role === "module" && systemOf(entry) === system && !pending(deleteKey(entry.path)))
  const others = tree.filter((entry) => (entry.role === "file" || (entry.role === "module" && !systemIds.has(systemOf(entry)))) && !pending(deleteKey(entry.path)))
  const titleOf = (entry: ArchitectureTreeEntry) => (entry.element ? elements.find((element) => element.id === entry.element)?.title : null)
    ?? (entry.role === "view" ? views.find((view) => view.id === entry.view)?.title : null)
    ?? entry.path.split("/").pop()!
  const act = async (write: () => Promise<ArchitectureWorkspace>, then: (next: ArchitectureWorkspace) => void) => {
    setProblem(null)
    try {
      then(await mutate(write))
    } catch (cause) {
      setProblem(failure(cause, "The architecture could not be changed"))
    }
  }
  const create = (title: string) => {
    if (!naming || !title) { setNaming(null); return }
    const op: ArchitectureOperation = naming.kind === "system" ? { op: "createSystem", title }
      : naming.kind === "module" ? { op: "createModule", system: naming.system, title }
      : { op: "createView", title }
    void act(() => applyArchitectureOperations(slug, [op], null), (next) => {
      setNaming(null)
      const [created] = next.created
      if (created) onOpen(created)
    })
  }
  const remove = (paths: string[], label: string) => {
    setProblem(null)
    schedule(deleteKey(paths[0]), label, async () => {
      await mutate(() => deleteArchitectureSources(slug, paths))
      if (paths.includes(currentRef.current)) onOpen("")
    })
  }
  const nameField = (placeholder: string, depth: number) => (
    <div className="py-1 pr-2" style={{ paddingLeft: 8 + depth * 12 }}>
      <input
        autoFocus placeholder={placeholder} aria-label={placeholder}
        className="w-full bg-transparent text-[11px] text-ink placeholder-ink-faint outline-none"
        onChange={() => setProblem(null)}
        onKeyDown={(event) => {
          if (event.key === "Enter") create(event.currentTarget.value.trim())
          if (event.key === "Escape") { event.stopPropagation(); setNaming(null) }
        }}
      />
    </div>
  )
  const row = (entry: ArchitectureTreeEntry, icon: ReactNode, depth: number, doomed: string[] = [entry.path]) => {
    const label = titleOf(entry)
    const here = entry.path === current
    return (
      <div key={entry.path}>
        <div className={cn("flex items-center pr-1", here && "bg-hover")}>
          <button type="button" onClick={() => onOpen(entry.path)} aria-current={here || undefined} className={cn("flex min-w-0 flex-1 items-center gap-1.5 py-1 text-left text-[11px] transition-colors", here ? "text-ink" : "text-ink-muted hover:text-ink")} style={{ paddingLeft: 8 + depth * 12 }}>
            {icon}<span className="truncate">{label}</span>
          </button>
          {here && <button type="button" aria-label={`Delete ${label}${doomed.length > 1 ? " and its modules" : ""}`} onClick={() => remove(doomed, label)} className="rounded p-0.5 text-ink-faint hover:text-danger transition-colors"><Trash2 className="h-3 w-3" /></button>}
        </div>
      </div>
    )
  }
  const icon = (Icon: typeof Network) => <Icon className="h-3 w-3 shrink-0 text-ink-faint" />
  const heading = (text: string) => <div className="px-2 pb-0.5 pt-2 text-[9px] uppercase tracking-wider text-ink-faint">{text}</div>
  const addButton = (label: string, onClick: () => void, depth = 0) => (
    <button type="button" onClick={() => { setProblem(null); setNaming(null); onClick() }} className="flex w-full items-center gap-1.5 py-1 text-left text-[11px] text-ink-faint hover:text-ink transition-colors" style={{ paddingLeft: 8 + depth * 12 }}>
      <Plus className="h-3 w-3 shrink-0" />{label}
    </button>
  )
  return (
    <nav aria-label="Architecture files" className="flex w-44 shrink-0 flex-col border-r border-divider/50">
      <div className="flex h-6 shrink-0 items-center border-b border-divider/50 pl-2 pr-1 text-[10px] text-ink-faint">
        <span className="flex-1">Architecture</span>
        <button type="button" aria-label="Hide files" onClick={onCollapse} className="rounded p-0.5 hover:text-ink"><PanelLeftClose className="h-3 w-3" /></button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto pb-2">
        {heading("Systems")}
        {systems.map((system) => {
          const id = system.element!
          const modules = modulesOf(id)
          const open = system.path === current || modules.some((module) => module.path === current)
          return (
            <div key={system.path}>
              {row(system, icon(Network), 0, [system.path, ...modules.map((module) => module.path)])}
              {modules.map((module) => row(module, icon(Box), 1))}
              {open && (naming?.kind === "module" && naming.system === id ? nameField("Module name", 1) : addButton("Module", () => setNaming({ kind: "module", system: id }), 1))}
            </div>
          )
        })}
        {naming?.kind === "system" ? nameField("System name", 0) : addButton("System", () => setNaming({ kind: "system" }))}
        {heading("Views")}
        {tree.filter((entry) => entry.role === "view" && !pending(deleteKey(entry.path))).map((entry) => row(entry, icon(Layers), 0))}
        {naming?.kind === "view" ? nameField("View name", 0) : addButton("View", () => setNaming({ kind: "view" }))}
        {others.length > 0 && heading("Other files")}
        {others.map((entry) => row(entry, icon(FileType), 0))}
        {problem && <p className="px-2 pt-1 text-[10px] text-danger" role="alert">{problem}</p>}
      </div>
    </nav>
  )
}

// Adds to a saved view from the whole model, grouped by the file declaring each
// element. "Add all" frames a file's systems with everything inside them.
function ArchitectureViewPicker({ elements, shown, onInclude }: {
  elements: ArchitectureElement[]
  shown: ArchitectureViewNode[]
  onInclude: (elements: string[], descendants: boolean) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useClickOutside(ref, useCallback(() => setOpen(false), []))
  const groups = useMemo(() => {
    const visible = new Set(shown.map((node) => node.id))
    const byFile = new Map<string, ArchitectureElement[]>()
    for (const element of elements) {
      if (!element.file) continue
      byFile.set(element.file, [...(byFile.get(element.file) ?? []), element])
    }
    return [...byFile.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([file, declared]) => {
      const here = new Set(declared.map((element) => element.id))
      return {
        file,
        roots: declared.filter((element) => !element.parent || !here.has(element.parent)).map((element) => element.id),
        rows: declared.sort((a, b) => a.id.localeCompare(b.id)).map((element) => ({ element, depth: element.id.split(".").length - 1, visible: visible.has(element.id) })),
      }
    })
  }, [elements, shown])
  const include = (ids: string[], descendants: boolean) => { onInclude(ids, descendants); setOpen(false) }
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button type="button" onClick={() => setOpen((current) => !current)} aria-haspopup="menu" aria-expanded={open} className="architecture-diagram-toolbar-symbol" style={{ width: "auto", padding: "0 6px", fontSize: 11 }}>
        <ListPlus size={13} /> Add to view
      </button>
      {open && (
        <div role="menu" aria-label="Add to view" style={{ position: "absolute", zIndex: 10, top: "calc(100% + 4px)", left: 0, width: 260, maxHeight: 320, overflowY: "auto", padding: 4, border: "1px solid var(--divider)", borderRadius: 6, background: "var(--paper)", boxShadow: "var(--shadow-lg)" }}>
          {groups.length === 0 && <p className="px-2 py-1.5 text-[11px] text-ink-faint">Nothing to add yet: create an architecture file first.</p>}
          {groups.map((group) => (
            <div key={group.file} className="py-0.5">
              <div className="flex items-center gap-2 px-2 py-1">
                <span className="min-w-0 flex-1 truncate text-[10px] uppercase tracking-wider text-ink-faint">{group.file}</span>
                <button type="button" role="menuitem" className="architecture-diagram-button" style={{ minHeight: 22 }} onClick={() => include(group.roots, true)}>Add all</button>
              </div>
              {group.rows.map(({ element, depth, visible }) => (
                <button key={element.id} type="button" role="menuitem" disabled={visible} onClick={() => include([element.id], false)} className="architecture-diagram-menu-item disabled:opacity-50" style={{ display: "flex", alignItems: "center", gap: 8, paddingLeft: 8 + depth * 12 }}>
                  <span className="min-w-0 flex-1 truncate">{element.title}</span>
                  <span className="shrink-0 text-[10px] text-ink-faint">{visible ? "shown" : element.kind}</span>
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
