import { useRef, type RefObject } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import { useTabActive } from "@/contexts/TabContext"
import { isEditableTarget, isModKey } from "@/lib/utils"
import type { DiagramGraph, DiagramNode } from "./diagramGraph"
import type { GraphFlowNode } from "./architectureFlow"
import { duplicateNodes } from "./graphEdits"

/** Ctrl+C, Ctrl+V and Ctrl+D on the surface's selected nodes, while `enabled` and the tab is shown. */
export function useNodeClipboard({ graphRef, nodes, emit, selectNodes, enabled }: {
  graphRef: RefObject<DiagramGraph>
  /** The flow nodes, whose `selected` flags name the selection. */
  nodes: readonly GraphFlowNode[]
  emit: (patch: Partial<DiagramGraph>) => void
  selectNodes: (ids: ReadonlySet<string>) => void
  enabled: boolean
}) {
  const active = useTabActive()
  // A snapshot, not ids: a paste repeats what was copied, even after the
  // originals are edited or deleted.
  const clipboardRef = useRef<DiagramNode[]>([])
  useEventListener("window", "keydown", (event) => {
    if (!isModKey(event)) return
    const key = event.key.toLowerCase()
    if (key !== "d" && key !== "c" && key !== "v") return
    if (isEditableTarget(event.target)) return
    const occupied = graphRef.current.nodes.map((node) => node.id)
    // Adds clones to the graph and selects them, not their sources, so repeated
    // Ctrl+D / Ctrl+V walks diagonally instead of stacking on the same spot.
    const addClones = (clones: DiagramNode[]) => {
      if (clones.length === 0) return
      emit({ nodes: [...graphRef.current.nodes, ...clones] })
      selectNodes(new Set(clones.map((clone) => clone.id)))
    }
    // The selected nodes as the graph has them.
    const selectedIds = new Set(nodes.filter((node) => node.selected).map((node) => node.id))
    const selected = graphRef.current.nodes.filter((node) => selectedIds.has(node.id))
    if (key === "d") {
      event.preventDefault()
      addClones(duplicateNodes(selected, occupied))
    } else if (key === "c") {
      if (selected.length === 0) return
      clipboardRef.current = selected
      event.preventDefault()
    } else if (clipboardRef.current.length > 0) {
      const clones = duplicateNodes(clipboardRef.current, occupied)
      // The next paste offsets from this one.
      clipboardRef.current = clones
      addClones(clones)
      event.preventDefault()
    }
  }, { active: enabled && active })
}
