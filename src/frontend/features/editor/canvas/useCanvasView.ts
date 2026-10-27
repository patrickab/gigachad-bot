"use client"

import { useCallback, useMemo, useState, type Dispatch, type SetStateAction } from "react"
import type { CanvasSelection } from "@/features/editor/canvas/CanvasWorkspace"
import type { CanvasToolbarSlots } from "@/features/editor/canvas/CanvasToolbar"
import { architectureSlug } from "@/lib/domain/architecturePaths"
import type { EdgeStyle } from "@/lib/domain/edgeRouting"
import { useTab } from "@/contexts/TabContext"

/** What a tab's canvas mode shows: one project's architecture (`graph/<slug>/<file>`, an empty file opens the first system), or a canvas. */
export type CanvasView =
  | { kind: "architecture"; path: string }
  | { kind: "canvas"; selection: CanvasSelection | null }

/** The sidebar props that pick what the canvas mode shows. */
interface CanvasSidebarProps {
  activeCanvasPath: string | null
  onCanvasSelect: (path: string, scope: CanvasSelection["scope"]) => void
  onCanvasDeleted: (path: string) => void
  onArchitectureSelect: (slug: string) => void
}

interface CanvasViewState {
  view: CanvasView
  sidebarProps: CanvasSidebarProps
  navigateArchitecture: (path: string) => void
  closeCanvas: () => void
  openCreatedCanvas: (selection: CanvasSelection) => void
  architectureEdgeStyle: EdgeStyle
  setArchitectureEdgeStyle: Dispatch<SetStateAction<EdgeStyle>>
  /** Top-bar elements the canvas toolbar renders into; null until both mount. */
  toolbarSlots: CanvasToolbarSlots | null
  setDrawingSlot: (element: HTMLElement | null) => void
  setViewSlot: (element: HTMLElement | null) => void
}

/** Own which canvas or architecture a tab's canvas mode shows; selecting one clears the other. */
export function useCanvasView(): CanvasViewState {
  const { tab, setWorkspaceItem } = useTab()
  const view = useMemo<CanvasView>(() => {
    const item = tab.workspaceItem
    if (item?.kind === "architecture") return { kind: "architecture", path: item.path }
    return { kind: "canvas", selection: item?.kind === "canvas" ? { path: item.path, scope: item.projectSlug ?? "" } : null }
  }, [tab.workspaceItem])
  const [architectureEdgeStyle, setArchitectureEdgeStyle] = useState<EdgeStyle>("elbow")
  const [drawingSlot, setDrawingSlot] = useState<HTMLElement | null>(null)
  const [viewSlot, setViewSlot] = useState<HTMLElement | null>(null)
  const toolbarSlots = useMemo(
    () => (drawingSlot && viewSlot ? { drawing: drawingSlot, view: viewSlot } : null),
    [drawingSlot, viewSlot],
  )

  const openCreatedCanvas = useCallback((selection: CanvasSelection) => {
    setWorkspaceItem({ kind: "canvas", path: selection.path, projectSlug: selection.scope || null })
  }, [setWorkspaceItem])
  const closeCanvas = useCallback(() => { setWorkspaceItem(null) }, [setWorkspaceItem])
  const navigateArchitecture = useCallback((path: string) => {
    setWorkspaceItem({ kind: "architecture", path, projectSlug: architectureSlug(path) })
  }, [setWorkspaceItem])

  const sidebarProps: CanvasSidebarProps = {
    activeCanvasPath: view.kind === "canvas" ? view.selection?.path ?? null : null,
    onCanvasSelect: (path, scope) => openCreatedCanvas({ path, scope }),
    onCanvasDeleted: (path) => {
      if (view.kind === "canvas" && view.selection?.path === path) closeCanvas()
    },
    onArchitectureSelect: (slug) => navigateArchitecture(`graph/${slug}/`),
  }

  return {
    view,
    sidebarProps,
    navigateArchitecture,
    closeCanvas,
    openCreatedCanvas,
    architectureEdgeStyle,
    setArchitectureEdgeStyle,
    toolbarSlots,
    setDrawingSlot,
    setViewSlot,
  }
}
