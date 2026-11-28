"use client"

import { useCallback, useEffect, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { LayoutGrid, ListTree } from "lucide-react"
import { writeArchitectureSource } from "@/lib/backend/api"
import { architectureSlug, architectureSource } from "@/lib/domain/architecturePaths"
import type { EdgeStyle } from "@/lib/domain/edgeRouting"
import { cn } from "@/lib/utils"
import { SEPARATOR, ToolButton, toolButton } from "@/features/editor/canvas/CanvasToolbar"
import { TopBarSlot, TopBarSlotProvider, useTopBarSlot } from "@/features/navigation/TopBarSlots"
import { TextDocumentEditor } from "@/features/editor/TextDocumentEditor"
import { ViewSwitch } from "@/components/ui/SegmentedSwitch"
import { ArchitectureDiagramSurface } from "./ArchitectureDiagramSurface"
import { ArchitectureTree } from "./ArchitectureTree"
import { ArchitectureViewPicker } from "./ArchitectureViewPicker"
import { RemoveReferenceDialog } from "./RemoveReferenceDialog"
import { useArchitectureView } from "./useArchitectureView"
import { useRemoveFromViewOffer } from "./useRemoveFromViewOffer"

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
  // With the tab's drawing and view slots the window renders its controls into the top bar and floats its file tree as a sidebar card; otherwise the controls sit inside the window.
  const drawingSlot = useTopBarSlot("drawing")
  const viewSlot = useTopBarSlot("view")
  const inTopBar = drawingSlot !== null && viewSlot !== null
  const { workspace, entry, view, graph, error, onChange, run, mutate } = useArchitectureView(slug, file)
  const open = useCallback((next: string) => onNavigate(`graph/${slug}/${next}`), [onNavigate, slug])
  // A new window, or one whose file was just deleted, opens the first system.
  useEffect(() => {
    if (file || !workspace) return
    const first = workspace.model.tree.find((candidate) => candidate.role === "system") ?? workspace.model.tree.find((candidate) => candidate.view)
    if (first) open(first.path)
  }, [file, workspace, open])
  const { removeFromView, offeredElements, answer } = useRemoveFromViewOffer({ run, view, workspace })
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
      {/* The save answers with the whole workspace, which replaces the shown one; a refused save rejects with the reason. */}
      <TextDocumentEditor key={file} path={`graph/${slug}/${file}`} slug={slug} onClose={() => setMode("diagram")}
        persistOverride={(content) => mutate(() => writeArchitectureSource(slug, file, content)).then(() => undefined)} />
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
        toolbar={toolbar} onRemoveFromView={landscape ? removeFromView : undefined}
        select={entry?.role === "module" ? entry.element : null} kinds={workspace.model.kinds} topLevelDrawing={!landscape}
      />
    )
  }
  const treeToggle = workspace && (
    <ToolButton icon={ListTree} label={treeOpen ? "Hide files" : "Show files"} active={treeOpen} aria-expanded={treeOpen} onClick={() => setTreeOpen((current) => !current)} />
  )
  const modeToggle = entry && (
    <ViewSwitch value={mode} onChange={setMode} options={[["diagram", "Diagram"], ["text", "Text"]]} />
  )
  return (
    // The surface's own slot scope: the window's mounting points inside the top-bar slots, so its controls keep a fixed order.
    <TopBarSlotProvider>
      <div className="flex h-full">
        {inTopBar && <>
          {createPortal(
            <div className="flex shrink-0 items-center gap-1">
              {treeToggle}
              <TopBarSlot name="drawing">{(ref) => <div ref={ref} className="flex items-center gap-1 empty:hidden before:mx-0.5 before:h-4 before:w-px before:bg-divider/50" />}</TopBarSlot>
            </div>,
            drawingSlot,
          )}
          {createPortal(
            <div className="flex shrink-0 items-center gap-1">
              {modeToggle}
              <TopBarSlot name="view">{(ref) => <div ref={ref} className="contents" />}</TopBarSlot>
            </div>,
            viewSlot,
          )}
        </>}
        {treeOpen && workspace && <ArchitectureTree slug={slug} workspace={workspace} current={file} onOpen={open} mutate={mutate} floating={inTopBar} />}
        <div className="flex min-w-0 flex-1 flex-col">
          {!inTopBar && (
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
          {offeredElements.length > 0 && <RemoveReferenceDialog elements={offeredElements} onAnswer={answer} />}
          {body}
        </div>
      </div>
    </TopBarSlotProvider>
  )
}
