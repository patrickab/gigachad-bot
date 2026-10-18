"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { LayoutGrid, ListPlus, ListTree, Trash2 } from "lucide-react"
import { applyArchitectureOperations, deleteArchitectureSources, readArchitecture } from "@/lib/backend/api"
import type {
  ArchitectureElement, ArchitectureOperation, ArchitectureTreeEntry, ArchitectureViewNode, ArchitectureWorkspace,
} from "@/lib/domain/architecture"
import { architectureSlug, architectureSource } from "@/lib/domain/architecturePaths"
import { diffGraph, fitGraph, graphFromView, type DiagramGraph, type DiagramNode } from "@/features/editor/architecture/diagramGraph"
import type { EdgeStyle } from "@/lib/domain/edgeRouting"
import { subscribeToChanges } from "@/lib/backend/syncStream"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { cn } from "@/lib/utils"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { useFileTree } from "@/components/ui/file-tree/useFileTree"
import { SEPARATOR, toolButton, type CanvasToolbarSlots } from "@/features/editor/canvas/CanvasToolbar"
import { ArchitectureDiagramSurface } from "./ArchitectureDiagramSurface"
import { ArchitectureSourceEditor } from "@/features/editor/TextDocumentEditor"
import { labelFontsReady, nodeMinWidth } from "./architectureLabelWidth"

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

  // Another file's project starts from nothing: no stale model, refusal, or load failure.
  const [shownSlug, setShownSlug] = useState(slug)
  if (shownSlug !== slug) {
    setShownSlug(slug)
    setWorkspace(null)
    setError(null)
    setLoadError(null)
  }
  useEffect(() => { refresh() }, [refresh])

  // Writes from this device are already reflected locally; anyone else's reload the model.
  useEffect(() => subscribeToChanges((event) => {
    if (event.resource_kind !== "document" || !event.resource_key.startsWith(`graph/${slug}/`)) return
    refresh()
  }), [slug, refresh])

  // How the tree lists `file`, and the view it opens in (a module opens its system's).
  const entry = useMemo(() => workspace?.model.tree.find((candidate) => candidate.path === file) ?? null, [workspace, file])
  const view = useMemo(() => workspace?.model.views.find((candidate) => candidate.id === entry?.view) ?? null, [workspace, entry])

  // Drafts belong to the view they were drawn on.
  const viewId = view?.id
  const [draftsViewId, setDraftsViewId] = useState(viewId)
  if (draftsViewId !== viewId) {
    setDraftsViewId(viewId)
    setDrafts([])
  }

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
  useLayoutEffect(() => { graphRef.current = graph }, [graph])

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

  return { workspace, entry, view, graph, error: error ?? loadError, onChange, run, mutate, refresh }
}

// An architecture window owns only its canvas-local frame, connection style, and
// the file it is on; the model lives in the project's LikeC4 workspace (every
// `.c4` file). The tree on its left moves between systems, modules, and saved
// views; the diagram is the view the file opens in (a module opens its system's,
// with the module selected), or the file itself as text. It is rendered at real
// layout size: CSS scaling would desynchronise React Flow's handles from the pointer.
export function ArchitectureWindow({ path, maximized, hostScale, edgeStyle = "elbow", onEdgeStyleChange, onNavigate, toolbarSlots }: {
  path: string
  maximized: boolean
  hostScale: number
  /** How connections are drawn: a window preference, LikeC4 has no such setting. */
  edgeStyle?: EdgeStyle
  onEdgeStyleChange: (style: EdgeStyle) => void
  /** Points the window at another `graph/<slug>/<file>`; an empty file means "the first system". */
  onNavigate: (path: string) => void
  /** App top-bar elements this window renders its controls into, and floats its file tree as a sidebar card; without them the controls sit inside the window. */
  toolbarSlots?: CanvasToolbarSlots | null
}) {
  const slug = architectureSlug(path)
  const file = architectureSource(path)
  const [treeOpen, setTreeOpen] = useState(true)
  const [mode, setMode] = useState<"diagram" | "text">("diagram")
  // The window's own mounting points inside the top-bar slots, so its controls keep a fixed order.
  const [surfaceDrawing, setSurfaceDrawing] = useState<HTMLElement | null>(null)
  const [surfaceView, setSurfaceView] = useState<HTMLElement | null>(null)
  const surfaceSlots = toolbarSlots && surfaceDrawing && surfaceView ? { drawing: surfaceDrawing, view: surfaceView } : null
  const { workspace, entry, view, graph, error, onChange, run, mutate, refresh } = useArchitectureView(slug, file)
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
  const [offeredViewId, setOfferedViewId] = useState(view?.id)
  if (offeredViewId !== view?.id) {
    setOfferedViewId(view?.id)
    setOffered([])
  }
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
  else if (!entry) body = notice(`${file} no longer exists.`)
  else if (mode === "text") body = (
    <div className="relative min-h-0 flex-1">
      {/* Saved text is this device's own write, which the change feed skips, so reread explicitly. */}
      <ArchitectureSourceEditor key={file} path={`graph/${slug}/${file}`} slug={slug} overlay onClose={() => setMode("diagram")} onSaved={refresh} />
    </div>
  )
  else if (!view || !graph) body = notice(`${file} declares no view yet.`, (
    <button type="button" className="architecture-diagram-button" onClick={() => run([{ op: "addFileView" }])}>Add a view of this file</button>
  ))
  else {
    // A landscape view shows what it names; a scoped one shows its system's surroundings.
    const landscape = view.editable && view.scope === null
    const toolbar = !view.editable
      ? <span className="text-[10px] text-ink-faint">Read-only: this view uses filters or styles, edit it as text</span>
      : <>
        {landscape && <>
          {SEPARATOR}
          <ArchitectureViewPicker elements={workspace.model.elements} shown={view.nodes} onInclude={(elements, descendants) => run([{ op: "includeInView", view: view.id, elements, descendants }])} />
        </>}
        {/* Hands every position back to LikeC4: arrows decide the ranks, the rest lines up. */}
        {view.manual && (
          <button type="button" className={cn(toolButton(false), "flex items-center gap-1 text-xs")}
            onClick={() => run([{ op: "layout", view: view.id, nodes: Object.fromEntries(view.nodes.map((node) => [node.id, null])), edges: Object.fromEntries(view.edges.map((edge) => [edge.id, null])) }])}>
            <LayoutGrid className="h-3.5 w-3.5" /> Auto layout
          </button>
        )}
      </>
    body = (
      <ArchitectureDiagramSurface
        key={view.id} graph={graph} onChange={onChange} readOnly={!view.editable} className="min-h-0 flex-1" autoFit={maximized} hostScale={hostScale}
        edgeStyle={edgeStyle} onEdgeStyleChange={onEdgeStyleChange}
        toolbar={toolbar} toolbarSlots={surfaceSlots} onRemoveFromView={landscape ? removeFromView : undefined}
        select={entry?.role === "module" ? entry.element : null} kinds={workspace.model.kinds} topLevelDrawing={!landscape}
      />
    )
  }
  const treeToggle = workspace && (
    <button type="button" aria-label={treeOpen ? "Hide files" : "Show files"} aria-expanded={treeOpen} onClick={() => setTreeOpen((current) => !current)} className={toolButton(treeOpen)}>
      <ListTree className="h-3.5 w-3.5" />
    </button>
  )
  const modeToggle = entry && (
    <div role="group" aria-label="Show as" className="flex gap-px rounded-md bg-surface p-0.5">
      {(["diagram", "text"] as const).map((option) => (
        <button key={option} type="button" aria-pressed={mode === option} onClick={() => setMode(option)} className={cn("rounded px-2 py-0.5 text-xs transition-colors", mode === option ? "bg-surface-elevated text-ink" : "text-ink-subtle hover:text-ink")}>
          {option === "diagram" ? "Diagram" : "Text"}
        </button>
      ))}
    </div>
  )
  return (
    <div className="flex h-full">
      {toolbarSlots && <>
        {createPortal(
          <div className="flex shrink-0 items-center gap-1">
            {treeToggle}
            <div ref={setSurfaceDrawing} className="flex items-center gap-1 empty:hidden before:mx-0.5 before:h-4 before:w-px before:bg-divider/50" />
          </div>,
          toolbarSlots.drawing,
        )}
        {createPortal(
          <div className="flex shrink-0 items-center gap-1">
            {modeToggle}
            <div ref={setSurfaceView} className="contents" />
          </div>,
          toolbarSlots.view,
        )}
      </>}
      {treeOpen && workspace && <ArchitectureTree slug={slug} workspace={workspace} current={file} onOpen={open} mutate={mutate} floating={!!toolbarSlots} />}
      <div className="flex min-w-0 flex-1 flex-col">
        {!toolbarSlots && (
          <div className="flex h-7 shrink-0 items-center gap-2 border-b border-divider/50 px-1.5 text-[10px] text-ink-faint">
            {treeToggle}
            <span className="min-w-0 flex-1 truncate">{file}</span>
            {modeToggle}
          </div>
        )}
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
function ArchitectureTree({ slug, workspace, current, onOpen, mutate, floating }: {
  slug: string
  workspace: ArchitectureWorkspace
  current: string
  onOpen: (file: string) => void
  mutate: Mutate
  /** A card like the app sidebar; otherwise a flat column inside a small window. */
  floating: boolean
}) {
  const { schedule, pending } = useUndoDelete()
  // Read when a delete commits, which may be after the user has moved to another file.
  const currentRef = useRef(current)
  useLayoutEffect(() => { currentRef.current = current }, [current])
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
  // Every system starts open, so its modules are visible.
  const systemPaths = allSystems.map((entry) => entry.path).join("\n")
  const defaultExpanded = useMemo(() => systemPaths.split("\n"), [systemPaths])
  const controller = useFileTree({ storageKey: "expanded_architecture_tree", defaultExpanded })
  // A refused name throws, which keeps the tree's name field open with the server's reason.
  const create = async (operation: ArchitectureOperation) => {
    const next = await mutate(() => applyArchitectureOperations(slug, [operation], null))
    const [created] = next.created
    if (created) onOpen(created)
  }
  const remove = (paths: string[], label: string) => {
    schedule(deleteKey(paths[0]), label, async () => {
      await mutate(() => deleteArchitectureSources(slug, paths))
      if (paths.includes(currentRef.current)) onOpen("")
    })
  }
  const itemOf = (entry: ArchitectureTreeEntry, doomed: string[] = [entry.path]): FileTreeItem<string> => ({
    id: entry.path, label: titleOf(entry), type: "element", isActive: entry.path === current,
    onSelect: () => onOpen(entry.path),
    onDelete: () => remove(doomed, titleOf(entry)),
  })
  const systemItems = systems.map((system): FileTreeItem<string> => {
    const modules = modulesOf(system.element!)
    return {
      ...itemOf(system, [system.path, ...modules.map((module) => module.path)]),
      type: "folder",
      children: modules.map((module) => itemOf(module)),
      create: { label: `New module in ${titleOf(system)}`, placeholder: "Module name", onSubmit: (title) => create({ op: "createModule", system: system.element!, title }) },
    }
  })
  const viewItems = tree.filter((entry) => entry.role === "view" && !pending(deleteKey(entry.path))).map((entry) => itemOf(entry))
  const separator = <div role="separator" className="my-4 border-t border-divider/50" />
  return (
    <nav aria-label="Architecture files" className={cn("flex shrink-0 flex-col", floating ? "mb-3 ml-3 w-60 rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]" : "w-44 border-r border-divider/50")}>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        <FileTree label="Systems" items={systemItems} controller={controller}
          create={{ label: "New system", placeholder: "System name", onSubmit: (title) => create({ op: "createSystem", title }) }} />
        {separator}
        <FileTree label="Views" items={viewItems} controller={controller}
          create={{ label: "New view", placeholder: "View name", onSubmit: (title) => create({ op: "createView", title }) }} />
        {others.length > 0 && <>
          {separator}
          <FileTree label="Other files" items={others.map((entry) => itemOf(entry))} controller={controller} />
        </>}
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
      <button type="button" onClick={() => setOpen((current) => !current)} aria-haspopup="menu" aria-expanded={open} className={cn(toolButton(open), "flex items-center gap-1 text-xs")}>
        <ListPlus className="h-3.5 w-3.5" /> Add to view
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
