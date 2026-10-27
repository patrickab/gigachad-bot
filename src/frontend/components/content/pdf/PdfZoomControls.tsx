"use client"

import { useState, type Dispatch, type RefObject, type SetStateAction } from "react"
import { flushSync } from "react-dom"
import { Minus, Moon, Plus, Sun } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { useMenuPopover } from "@/components/ui/MenuPopover"
import { useEventListener } from "@/hooks/useEventListener"
import { switchColors } from "@/lib/themeTransition"
import { MAX_ZOOM, MIN_ZOOM, type PdfZoom } from "./usePdfZoom"

const CONTROLS_PADDING = 12
// controls show while the pointer is within this many px of them
const CONTROLS_VICINITY = 80
const ZOOM_PRESETS = [1, 1.25, 1.5, 1.75, 2, 2.5]

/** Zoom out / preset menu / zoom in, plus the light-paper toggle, CONTROLS_PADDING px inside the page's top-right corner.
 *
 * - Shows while the pointer moves within CONTROLS_VICINITY px of it over `hostRef`, or while the zoom menu is open
 */
export function PdfZoomControls({ hostRef, zoom, topInset, pageRightGap, lightPaper, setLightPaper }: {
  hostRef: RefObject<HTMLElement | null>
  zoom: PdfZoom
  topInset: number
  /** Px between the page's right edge and the viewer's. */
  pageRightGap: number
  lightPaper: boolean
  setLightPaper: Dispatch<SetStateAction<boolean>>
}) {
  const { ref: menuRef, open: menuOpen, toggle: toggleMenu, close: closeMenu } = useMenuPopover({ escape: false })
  const [nearControls, setNearControls] = useState(false)
  useEventListener(hostRef, "mousemove", (event) => {
    const rect = menuRef.current?.getBoundingClientRect()
    if (!rect) return
    const dx = Math.max(rect.left - event.clientX, 0, event.clientX - rect.right)
    const dy = Math.max(rect.top - event.clientY, 0, event.clientY - rect.bottom)
    setNearControls(Math.hypot(dx, dy) <= CONTROLS_VICINITY)
  })
  useEventListener(hostRef, "mouseleave", () => setNearControls(false))
  const visible = nearControls || menuOpen
  return (
    <div
      ref={menuRef}
      style={{ top: topInset + CONTROLS_PADDING, right: Math.max(0, pageRightGap) + CONTROLS_PADDING }}
      className={`absolute z-10 flex items-center gap-1 rounded-lg bg-surface-elevated/90 backdrop-blur-sm border border-divider/60 shadow-[var(--shadow-md)] transition-opacity duration-300 pointer-events-auto ${visible ? "opacity-100" : "opacity-0 pointer-events-none"}`}
    >
      <button aria-label="Zoom out"
        onClick={() => zoom.step(-1)}
        disabled={zoom.zoom <= MIN_ZOOM}
        className="rounded-l-lg p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:text-ink-faint disabled:pointer-events-none transition-colors"
      >
        <Minus className="h-3.5 w-3.5" strokeWidth={2} />
      </button>
      <div className="relative">
        <button
          onClick={toggleMenu}
          className="px-1.5 py-1.5 text-[10px] font-medium tabular-nums text-ink-muted hover:text-ink hover:bg-hover transition-colors select-none"
        >
          {`${Math.round(zoom.zoom * 100)}%`}
        </button>
        <AnimatePresence>
          {menuOpen && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.12 }}
              className="absolute top-full left-1/2 -translate-x-1/2 mt-1 py-1 rounded-lg bg-surface-elevated border border-divider shadow-[var(--shadow-lg)] z-20"
            >
              {ZOOM_PRESETS.map((preset) => (
                <button
                  key={preset}
                  onClick={() => {
                    zoom.zoomAt(preset)
                    closeMenu()
                  }}
                  className={`block w-full px-4 py-1 text-[10px] font-medium tabular-nums text-left hover:bg-hover transition-colors ${
                    Math.abs(zoom.zoom - preset) < 0.01 ? "text-ink" : "text-ink-muted"
                  }`}
                >
                  {Math.round(preset * 100)}%
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      <button aria-label="Zoom in"
        onClick={() => zoom.step(1)}
        disabled={zoom.zoom >= MAX_ZOOM}
        className="p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:text-ink-faint disabled:pointer-events-none transition-colors in-[.light]:rounded-r-lg"
      >
        <Plus className="h-3.5 w-3.5" strokeWidth={2} />
      </button>
      <div className="pdf-paper-toggle flex items-center">
        <div className="w-px h-4 bg-divider" />
        <button
          aria-label={lightPaper ? "Dark PDF pages" : "Light PDF pages"}
          onClick={() => switchColors(() => flushSync(() => setLightPaper((light) => !light)))}
          className="rounded-r-lg p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
        >
          {lightPaper ? <Moon className="h-3.5 w-3.5" /> : <Sun className="h-3.5 w-3.5" />}
        </button>
      </div>
    </div>
  )
}
