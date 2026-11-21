"use client"

import { useCallback, useEffect, useState, type RefObject } from "react"
import type { PdfPageBox } from "@/components/content/PdfViewer"
import { RENDER_WIDTH } from "@/components/content/pdf/pdfLayout"
import { useElementSize } from "@/hooks/useElementSize"
import { Columns2, FileText, Terminal } from "lucide-react"
import { PillTabs } from "@/components/ui/PillTabs"
import { readStorage, writeStorage } from "@/lib/storage"

/** What the editor and rendered file split shows: one pane or both side by side. */
export type WorkbenchView = "editor" | "split" | "render"

/** Where the editor and the rendered file sit in their panes: same width, same top edge, `height` is the editor's. */
export type PaneLayout = PdfPageBox & { height: number }

const STORAGE_KEY = "workbench_view"

/** One of the split's two panes: Neovim's page or the rendered file. */
export type WorkbenchSide = "editor" | "render"

/** The view, remembered on this device for every workbench tab, and `show(side)`, which brings `side` back when the view hides it. */
export function useWorkbenchView(): [WorkbenchView, (view: WorkbenchView) => void, (side: WorkbenchSide) => void] {
  const [view, setView] = useState<WorkbenchView>(() => {
    const stored = readStorage(STORAGE_KEY)
    return stored === "editor" || stored === "render" ? stored : "split"
  })
  useEffect(() => { writeStorage(STORAGE_KEY, view) }, [view])
  // Reads the view it changes, so a call from an answer that arrived later sees the view of that moment
  const show = useCallback((side: WorkbenchSide) => setView((current) => current === "split" || current === side ? current : "split"), [])
  return [view, setView, show]
}

/** Height of the floating top bar (`TopBar`, h-11) the panes sit beneath. */
export const TOP_BAR_PX = 44
/** Space around each pane's content. With the split area's own 12 px padding every gap is 24 px, twice the sidebar's inset. */
const MARGIN_X = 12
const MARGIN_Y = 12
/**
 * Where both panes' boxes start (below the top bar and the margin) and how far above the bottom they end.
 * A framed surface, unlike an endless one (the canvas), keeps the same space above and below its content.
 */
const PANE_INSET_Y_PX = TOP_BAR_PX + MARGIN_Y

/**
 * Where Neovim and the rendered file sit in a pane of the split in `areaRef`: same width, same top edge.
 *
 * - Width fills the pane up to the PDF viewer's `RENDER_WIDTH` cap, centered, so opening a side panel only changes the width
 * - `height` ends as far above the bottom as the panes start below the top, the rendered pane scrolls inside it
 */
export function usePaneLayout(areaRef: RefObject<HTMLElement | null>, view: WorkbenchView): PaneLayout {
  const area = useElementSize(areaRef)
  const paneWidth = view === "split" ? area.width / 2 : area.width
  const width = Math.max(0, Math.floor(Math.min(RENDER_WIDTH, paneWidth - 2 * MARGIN_X)))
  return { width, top: PANE_INSET_Y_PX, bottom: PANE_INSET_Y_PX, height: Math.max(0, area.height - 2 * PANE_INSET_Y_PX) }
}

const VIEW_OPTIONS = [
  { value: "editor", label: <Terminal className="h-3.5 w-3.5" />, ariaLabel: "Editor only" },
  { value: "split", label: <Columns2 className="h-3.5 w-3.5" />, ariaLabel: "Split view" },
  { value: "render", label: <FileText className="h-3.5 w-3.5" />, ariaLabel: "Preview only" },
] as const

/** Top-bar switch between editor only, both side by side, and the rendered file only. */
export function WorkbenchViewSwitch({ view, onChange }: { view: WorkbenchView; onChange: (view: WorkbenchView) => void }) {
  return <PillTabs options={VIEW_OPTIONS} value={view} onChange={onChange} pillClassName="px-2" />
}
