"use client"

import { useState, type RefObject } from "react"
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

/** The view, remembered on this device for every workbench tab. */
export function useWorkbenchView(): [WorkbenchView, (view: WorkbenchView) => void] {
  const [view, setView] = useState<WorkbenchView>(() => {
    const stored = readStorage(STORAGE_KEY)
    return stored === "editor" || stored === "render" ? stored : "split"
  })
  const change = (next: WorkbenchView) => {
    setView(next)
    writeStorage(STORAGE_KEY, next)
  }
  return [view, change]
}

/** Height of the floating top bar (`TopBar`, h-11) the panes sit beneath. */
export const TOP_BAR_PX = 44
/** Space around each pane's content. With the split area's own 12 px padding every gap is 24 px, twice the sidebar's inset. */
const MARGIN_X = 12
const MARGIN_Y = 12
/** Where a pane's content starts: below the top bar and the margin. The rendered pages and the notebook scroll on underneath the bar. */
export const PANE_TOP_PX = TOP_BAR_PX + MARGIN_Y

/**
 * Where Neovim and the rendered file sit in a pane of the split in `areaRef`: same width, same top edge.
 *
 * - Width fills the pane up to the PDF viewer's `RENDER_WIDTH` cap, centered, so opening a side panel only changes the width
 * - The rendered pages scroll on below, `height` is the editor's: down to the bottom margin
 */
export function usePaneLayout(areaRef: RefObject<HTMLElement | null>, view: WorkbenchView): PaneLayout {
  const area = useElementSize(areaRef)
  const paneWidth = view === "split" ? area.width / 2 : area.width
  const width = Math.max(0, Math.floor(Math.min(RENDER_WIDTH, paneWidth - 2 * MARGIN_X)))
  const top = PANE_TOP_PX
  return { width, top, bottom: MARGIN_Y, height: Math.max(0, area.height - top - MARGIN_Y) }
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
