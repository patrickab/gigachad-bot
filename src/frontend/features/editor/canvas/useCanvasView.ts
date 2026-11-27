"use client"

import { useCallback, useMemo, useState, type Dispatch, type SetStateAction } from "react"
import type { CanvasSelection } from "@/features/editor/canvas/CanvasWorkspace"
import { architectureSlug } from "@/lib/domain/architecturePaths"
import type { EdgeStyle } from "@/lib/domain/edgeRouting"
import { scratchItem, shownItem, useTab } from "@/contexts/TabContext"

/** What a tab's canvas mode shows: one project's architecture (`graph/<slug>/<file>`, an empty file opens the first system), or a canvas. */
export type CanvasView =
  | { kind: "architecture"; path: string }
  | { kind: "canvas"; selection: CanvasSelection | null }

interface CanvasViewState {
  view: CanvasView
  navigateArchitecture: (path: string) => void
  closeCanvas: () => void
  openCreatedCanvas: (selection: CanvasSelection) => void
  architectureEdgeStyle: EdgeStyle
  setArchitectureEdgeStyle: Dispatch<SetStateAction<EdgeStyle>>
}

/** Own which canvas or architecture a tab's canvas mode shows; selecting one clears the other. */
export function useCanvasView(): CanvasViewState {
  const { tab, setWorkspaceItem } = useTab()
  const item = shownItem(tab)
  const view = useMemo<CanvasView>(() => {
    if (item?.kind === "architecture") return { kind: "architecture", path: item.path }
    return { kind: "canvas", selection: item?.kind === "canvas" ? { path: item.path, scope: item.projectSlug ?? "" } : null }
  }, [item])
  const [architectureEdgeStyle, setArchitectureEdgeStyle] = useState<EdgeStyle>("elbow")

  const openCreatedCanvas = useCallback((selection: CanvasSelection) => {
    setWorkspaceItem({ kind: "canvas", path: selection.path, projectSlug: selection.scope || null })
  }, [setWorkspaceItem])
  const closeCanvas = useCallback(() => { setWorkspaceItem(scratchItem(tab.projectSlug)) }, [setWorkspaceItem, tab.projectSlug])
  const navigateArchitecture = useCallback((path: string) => {
    setWorkspaceItem({ kind: "architecture", path, projectSlug: architectureSlug(path) })
  }, [setWorkspaceItem])

  return {
    view,
    navigateArchitecture,
    closeCanvas,
    openCreatedCanvas,
    architectureEdgeStyle,
    setArchitectureEdgeStyle,
  }
}
