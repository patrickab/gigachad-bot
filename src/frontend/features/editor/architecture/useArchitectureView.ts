import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { applyArchitectureOperations, readArchitecture } from "@/lib/backend/api"
import type { ArchitectureOperation, ArchitectureWorkspace } from "@/lib/domain/architecture"
import { subscribeToChanges, syncCursorReady } from "@/lib/backend/syncStream"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { errorMessage } from "@/lib/utils"
import { diffGraph, fitGraph, graphFromView, type DiagramGraph, type DiagramNode } from "./diagramGraph"
import { labelFontsReady, nodeMinWidth } from "./architectureLabelWidth"

export type Mutate = (write: () => Promise<ArchitectureWorkspace>) => Promise<ArchitectureWorkspace>

/**
 * A `.c4` file of a project's architecture, drawn through the view it opens
 * in: a system's own `view x of x`, its system's view for a module, or a
 * saved view. Reads and writes of the model run one at a time in order; each
 * write is validated by the server against the model as it is then, so a
 * stale edit is refused with a reason instead of overwriting newer work.
 */
export function useArchitectureView(slug: string, file: string) {
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
      // After the change feed holds its head, so a change racing this read still arrives as an event.
      return syncCursorReady().then(() => readArchitecture(slug))
    }).then(() => setLoadError(null), (cause: unknown) => setLoadError(errorMessage(cause, "Could not load the architecture")))
  }, [mutate, slug])

  // Another file's project starts from nothing: no stale model, refusal, or load failure.
  useResetOnChange(slug, () => {
    setWorkspace(null)
    setError(null)
    setLoadError(null)
  })
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
  useResetOnChange(view?.id, () => setDrafts([]))

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
    mutate(write).catch((cause: unknown) => setError(errorMessage(cause, "The architecture could not be changed")))
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
