"use client"

import { useEffect, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { Camera, CircleDashed, Redo2, Undo2 } from "lucide-react"
import { PRESET_COLORS, inkColor, type StrokeSize } from "@/features/editor/canvas/ink/canvasInk"
import { cn } from "@/lib/utils"

/** Header elements the canvas toolbar portals into: drawing tools left, view and history controls right. */
export type CanvasToolbarSlots = { drawing: HTMLElement; view: HTMLElement }

/** The pointer tool: `draw` is the default; select (lasso) and screenshot are modal and exclusive. */
export type CanvasTool = "draw" | "select" | "screenshot"

const RECENT_COLORS = 3
const SEPARATOR = <div className="w-px h-4 bg-divider/50 mx-0.5" />

/** Preset, recent and custom ink colours behind a popover. */
function ColorPicker({ color, isDark, onChange }: { color: string; isDark: boolean; onChange: (color: string) => void }) {
  const [open, setOpen] = useState(false)
  const [recent, setRecent] = useState<string[]>([])
  const ref = useRef<HTMLDivElement>(null)
  // close on outside click, skip if the native picker is active
  const nativePickerOpen = useRef(false)

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (nativePickerOpen.current) return
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [open])

  const pick = (c: string, close = true) => {
    onChange(c)
    setRecent((prev) => [c, ...prev.filter((x) => x !== c)].slice(0, RECENT_COLORS))
    if (close) setOpen(false)
  }

  const swatch = (c: string) => (
    <button
      key={c}
      onClick={() => pick(c)}
      className={cn(
        "w-5 h-5 rounded-full border-2 transition-all",
        color === c ? "border-ink scale-110" : "border-transparent hover:border-ink-faint"
      )}
      style={{ backgroundColor: inkColor(c, isDark) }}
    />
  )

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 rounded px-2 py-1 text-[10px] font-medium hover:bg-hover transition-colors"
        style={{ color: inkColor(color, isDark) }}
      >
        Color
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 z-10 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] p-2 min-w-[100px]">
          <div className="grid grid-cols-4 gap-1 mb-1.5">{PRESET_COLORS.map((c) => swatch(c.value))}</div>
          {recent.length > 0 && (
            <>
              <div className="text-[9px] text-ink-faint uppercase tracking-wider mb-1">Recent</div>
              <div className="flex gap-1 mb-1.5">{recent.map(swatch)}</div>
            </>
          )}
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="color"
              value={color}
              onFocus={() => { nativePickerOpen.current = true }}
              onBlur={() => { nativePickerOpen.current = false }}
              onChange={(e) => pick(e.target.value, false)}
              className="w-5 h-5 rounded border-0 p-0 cursor-pointer bg-transparent [&::-webkit-color-swatch-wrapper]:p-0 [&::-webkit-color-swatch]:rounded [&::-webkit-color-swatch]:border-divider"
            />
            <span className="text-[10px] text-ink-subtle">Custom</span>
          </label>
        </div>
      )}
    </div>
  )
}

interface CanvasToolbarProps {
  /** When given, the toolbar renders into these external elements (the app header pills in canvas mode) instead of above the drawing surface. */
  slots?: CanvasToolbarSlots | null
  /** The "+" popover. */
  addMenu: ReactNode
  strokeSize: StrokeSize
  onCycleStrokeSize: () => void
  color: string
  onColorChange: (color: string) => void
  isDark: boolean
  tool: CanvasTool
  onToggleTool: (tool: Exclude<CanvasTool, "draw">) => void
  isErasing: boolean
  selectedCount: number
  onUndo: () => void
  onRedo: () => void
  canUndo: boolean
  canRedo: boolean
  zoomPercent: number
}

const toolButton = (active: boolean) => cn("rounded p-1 transition-colors", active ? "text-ink bg-hover" : "text-ink-subtle hover:text-ink hover:bg-hover")

/** Drawing tools on the left, view and history controls on the right. */
export function CanvasToolbar({
  slots, addMenu, strokeSize, onCycleStrokeSize, color, onColorChange, isDark, tool, onToggleTool, isErasing, selectedCount,
  onUndo, onRedo, canUndo, canRedo, zoomPercent,
}: CanvasToolbarProps) {
  const drawingTools = (
    <div className="flex items-center gap-1 shrink-0 min-w-0">
      {addMenu}
      {SEPARATOR}
      <button aria-label="Change pen width"
        onClick={onCycleStrokeSize}
        className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
      >
        <svg width="10" height="10" viewBox="0 0 10 10" className="shrink-0">
          <circle cx="5" cy="5" r={strokeSize === "thin" ? 1.5 : strokeSize === "medium" ? 2.5 : 4} fill="currentColor" />
        </svg>
      </button>
      {SEPARATOR}
      <ColorPicker color={color} isDark={isDark} onChange={onColorChange} />
      {SEPARATOR}
      <button aria-label="Select on canvas" onClick={() => onToggleTool("select")} className={toolButton(tool === "select")}>
        <CircleDashed className="h-3.5 w-3.5" />
      </button>
      {isErasing && <span className="text-[10px] text-ink-faint ml-1">(eraser)</span>}
      {tool === "select" && <span className="text-[10px] text-ink-faint ml-1">(draw a lasso to select)</span>}
      {selectedCount > 0 && <span className="text-[10px] text-ink-faint ml-1">({selectedCount} selected — drag, resize or Esc)</span>}
    </div>
  )

  const viewTools = (
    <div className="flex items-center gap-1 shrink-0">
      {tool === "screenshot" && <span className="text-[10px] text-ink-faint mr-1">(drag to capture)</span>}
      <button aria-label="Capture canvas" onClick={() => onToggleTool("screenshot")} className={toolButton(tool === "screenshot")}>
        <Camera className="h-3.5 w-3.5" />
      </button>
      {SEPARATOR}
      <button aria-label="Undo"
        onClick={onUndo}
        disabled={!canUndo}
        className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
      >
        <Undo2 className="h-3.5 w-3.5" />
      </button>
      {SEPARATOR}
      <button aria-label="Redo"
        onClick={onRedo}
        disabled={!canRedo}
        className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
      >
        <Redo2 className="h-3.5 w-3.5" />
      </button>
      {SEPARATOR}
      <span className="text-[10px] text-ink-faint tabular-nums">{zoomPercent}%</span>
    </div>
  )

  // Inline above the canvas, or portaled into the app header pills in canvas mode
  if (slots) return <>{createPortal(drawingTools, slots.drawing)}{createPortal(viewTools, slots.view)}</>
  return (
    <div className="flex items-center gap-1 shrink-0 min-w-0 px-2 py-1.5 border-b border-divider/50">
      {drawingTools}
      <div className="flex-1" />
      {viewTools}
    </div>
  )
}
