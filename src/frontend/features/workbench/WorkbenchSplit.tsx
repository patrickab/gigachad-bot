"use client"

import { useState, type RefObject } from "react"
import type { PdfPageBox } from "@/components/content/PdfViewer"
import { RENDER_WIDTH } from "@/components/content/pdf/pdfLayout"
import { useElementSize } from "@/hooks/useElementSize"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { readStorage, writeStorage } from "@/lib/storage"
import { cn } from "@/lib/utils"

/** Which pane of the editor and rendered file split is hidden, null shows both. */
export type HiddenPane = "editor" | "render" | null

/** Where the editor and the rendered file sit in their panes: same width, same top edge, `height` is the editor's. */
export type PaneLayout = PdfPageBox & { height: number }

const STORAGE_KEY = "workbench_hidden_pane"

/** The hidden pane, remembered on this device for every workbench tab. */
export function useHiddenPane(): [HiddenPane, (pane: HiddenPane) => void] {
  const [hidden, setHidden] = useState<HiddenPane>(() => {
    const stored = readStorage(STORAGE_KEY)
    return stored === "editor" || stored === "render" ? stored : null
  })
  const change = (pane: HiddenPane) => {
    setHidden(pane)
    writeStorage(STORAGE_KEY, pane ?? "")
  }
  return [hidden, change]
}

/** Height of the floating top bar (`TopBar`, h-11) the panes sit beneath. */
export const TOP_BAR_PX = 44
/** Space around each pane's content. With the split area's own 12 px padding every gap is 24 px, twice the sidebar's inset. */
const MARGIN_X = 12
const MARGIN_Y = 12

/**
 * Where Neovim and the rendered file sit in a pane of the split in `areaRef`: same width, same top edge.
 *
 * - Width fills the pane up to the PDF viewer's `RENDER_WIDTH` cap, centered, so opening a side panel only changes the width
 * - The rendered pages scroll on below, `height` is the editor's: down to the bottom margin
 */
export function usePaneLayout(areaRef: RefObject<HTMLElement | null>, hidden: HiddenPane): PaneLayout {
  const area = useElementSize(areaRef)
  // The split line takes 1 px between two panes
  const paneWidth = hidden ? area.width : (area.width - 1) / 2
  const width = Math.max(0, Math.floor(Math.min(RENDER_WIDTH, paneWidth - 2 * MARGIN_X)))
  const top = TOP_BAR_PX + MARGIN_Y
  return { width, top, bottom: MARGIN_Y, height: Math.max(0, area.height - top - MARGIN_Y) }
}

const BUTTON = "pointer-events-auto rounded-full border border-divider/50 bg-surface p-0.5 text-ink-subtle shadow-sm hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/**
 * The line between Neovim and the rendered file. Hovering it offers to hide either side, with one side hidden it brings it back.
 *
 * - Buttons point the way the line moves: left hides the editor, right hides the rendered file
 * - `renderLabel` names the rendered file in the button labels ("PDF": "Show PDF", "Hide PDF")
 */
export function WorkbenchSplit({ hidden, onChange, renderLabel }: { hidden: HiddenPane; onChange: (pane: HiddenPane) => void; renderLabel: string }) {
  return (
    <div className="relative mt-11 w-px shrink-0 bg-divider/40">
      <div className="group absolute inset-y-0 -left-3 -right-3 z-10 flex flex-col items-center justify-center gap-1">
        <div className={cn("flex flex-col gap-1 transition-opacity", hidden ? "opacity-60 group-hover:opacity-100" : "opacity-0 group-hover:opacity-100 focus-within:opacity-100")}>
          {hidden !== "editor" && (
            <button type="button" aria-label={hidden === "render" ? `Show ${renderLabel}` : "Hide editor"} onClick={() => onChange(hidden === "render" ? null : "editor")} className={BUTTON}>
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
          )}
          {hidden !== "render" && (
            <button type="button" aria-label={hidden === "editor" ? "Show editor" : `Hide ${renderLabel}`} onClick={() => onChange(hidden === "editor" ? null : "render")} className={BUTTON}>
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/** In the top bar while a pane is hidden: one click brings it back. */
export function ShowPaneButton({ hidden, onChange, renderLabel }: { hidden: HiddenPane; onChange: (pane: HiddenPane) => void; renderLabel: string }) {
  if (!hidden) return null
  return (
    <button type="button" onClick={() => onChange(null)} className="rounded px-1 text-[11px] text-ink-subtle transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
      {hidden === "render" ? `Show ${renderLabel}` : "Show editor"}
    </button>
  )
}
