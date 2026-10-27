"use client"

import { useCallback, useMemo, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { Camera, CircleDashed, Eraser, Highlighter, PenLine, Redo2, Undo2, type LucideIcon } from "lucide-react"
import { DEFAULT_PEN_COLOR, HIGHLIGHT_COLORS, PRESET_COLORS, inkColor, strokeWidth, type InkTool, type StrokeSize } from "@/features/editor/canvas/ink/canvasInk"
import { useOutsidePress } from "@/hooks/useOutsidePress"
import { cn } from "@/lib/utils"

/** Header elements the canvas toolbar portals into: drawing tools left, view and history controls right. */
export type CanvasToolbarSlots = { drawing: HTMLElement; view: HTMLElement }

/** The pointer tool: `draw` is the default; select (lasso) and screenshot are modal and exclusive. */
export type CanvasTool = "draw" | "select" | "screenshot"

const RECENT_COLORS = 3
export const SEPARATOR = <div className="w-px h-4 bg-divider/50 mx-0.5" />

/** Preset, recent and custom ink colours behind a popover. */
function ColorPicker({ color, onChange, presets = PRESET_COLORS }: {
  color: string
  onChange: (color: string) => void
  presets?: ReadonlyArray<{ value: string }>
}) {
  const [open, setOpen] = useState(false)
  const [recent, setRecent] = useState<string[]>([])
  const ref = useRef<HTMLDivElement>(null)
  // close on outside click, skip if the native picker is active
  const nativePickerOpen = useRef(false)

  useOutsidePress(ref, () => { if (!nativePickerOpen.current) setOpen(false) }, { active: open })

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
      style={{ backgroundColor: inkColor(c) }}
    />
  )

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 rounded px-2 py-1 text-[10px] font-medium hover:bg-hover transition-colors"
        style={{ color: inkColor(color) }}
      >
        Color
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 z-10 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] p-2 min-w-[100px]">
          <div className="grid grid-cols-4 gap-1 mb-1.5">{presets.map((c) => swatch(c.value))}</div>
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

// the size button steps through the widths
const NEXT_SIZE: Record<StrokeSize, StrokeSize> = { thin: "medium", medium: "thick", thick: "thin" }

/** Pen and highlighter settings behind `InkTools`. `color` and `width` are what a new stroke gets. */
interface InkSettings {
  tool: InkTool | null
  toggleTool: (tool: InkTool) => void
  clearTool: () => void
  size: StrokeSize
  cycleSize: () => void
  penColor: string
  highlighting: boolean
  color: string
  width: number
  setColor: (color: string) => void
}

/** Ink tool state: pen and highlighter keep their own colour, the size applies to both. */
export function useInkSettings(): InkSettings {
  const [tool, setTool] = useState<InkTool | null>(null)
  const [size, setSize] = useState<StrokeSize>("thin")
  const [penColor, setPenColor] = useState(DEFAULT_PEN_COLOR)
  const [highlightColor, setHighlightColor] = useState<string>(HIGHLIGHT_COLORS[0].value)
  const highlighting = tool === "highlighter"
  const toggleTool = useCallback((next: InkTool) => setTool((current) => current === next ? null : next), [])
  const clearTool = useCallback(() => setTool(null), [])
  const cycleSize = useCallback(() => setSize((current) => NEXT_SIZE[current]), [])
  return useMemo(() => ({
    tool, toggleTool, clearTool, size, cycleSize, penColor, highlighting,
    color: highlighting ? highlightColor : penColor,
    width: strokeWidth(size, highlighting),
    setColor: highlighting ? setHighlightColor : setPenColor,
  }), [tool, toggleTool, clearTool, size, cycleSize, penColor, highlightColor, highlighting])
}

/** Pen, highlighter and eraser, stroke width and the selected tool's colour. */
export function InkTools({ ink }: { ink: InkSettings }) {
  return (
    <>
      <ToolButton icon={PenLine} label="Pen" toggle active={ink.tool === "pen"} onClick={() => ink.toggleTool("pen")} />
      <ToolButton icon={Highlighter} label="Highlighter" toggle active={ink.highlighting} onClick={() => ink.toggleTool("highlighter")} />
      <ToolButton icon={Eraser} label="Eraser" toggle active={ink.tool === "eraser"} onClick={() => ink.toggleTool("eraser")} />
      {SEPARATOR}
      <button aria-label="Change stroke width" onClick={ink.cycleSize} className={toolButton(false)}>
        <svg width="10" height="10" viewBox="0 0 10 10" className="shrink-0">
          <circle cx="5" cy="5" r={ink.size === "thin" ? 1.5 : ink.size === "medium" ? 2.5 : 4} fill="currentColor" />
        </svg>
      </button>
      <ColorPicker color={ink.color} onChange={ink.setColor} presets={ink.highlighting ? HIGHLIGHT_COLORS : undefined} />
    </>
  )
}

interface CanvasToolbarProps {
  /** When given, the toolbar renders into these external elements (the app header pills in canvas mode) instead of above the drawing surface. */
  slots?: CanvasToolbarSlots | null
  /** The "+" popover. */
  addMenu: ReactNode
  ink: InkSettings
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

export const toolButton = (active: boolean) => cn("rounded p-1 transition-colors", active ? "text-ink bg-hover" : "text-ink-subtle hover:text-ink hover:bg-hover")

/** Icon-only toolbar button. `active` styles it as the selected tool, `toggle` also reports that as aria-pressed. */
export function ToolButton({ icon: Icon, label, active = false, toggle = false, ...props }: {
  icon: LucideIcon
  label: string
  active?: boolean
  toggle?: boolean
} & Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label" | "aria-pressed" | "className" | "children">) {
  return (
    <button type="button" aria-label={label} aria-pressed={toggle ? active : undefined} className={toolButton(active)} {...props}>
      <Icon className="h-3.5 w-3.5" />
    </button>
  )
}

/** Undo and Redo, with `separator` between them. */
export function UndoRedoButtons({ onUndo, onRedo, canUndo, canRedo, separator }: {
  onUndo: () => void
  onRedo: () => void
  canUndo: boolean
  canRedo: boolean
  separator?: ReactNode
}) {
  const historyButton = (label: string, Icon: LucideIcon, onClick: () => void, enabled: boolean) => (
    <button aria-label={label} onClick={onClick} disabled={!enabled} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors">
      <Icon className="h-3.5 w-3.5" />
    </button>
  )
  return <>{historyButton("Undo", Undo2, onUndo, canUndo)}{separator}{historyButton("Redo", Redo2, onRedo, canRedo)}</>
}

/** Drawing tools on the left, view and history controls on the right. */
export function CanvasToolbar({
  slots, addMenu, ink, tool, onToggleTool, isErasing, selectedCount,
  onUndo, onRedo, canUndo, canRedo, zoomPercent,
}: CanvasToolbarProps) {
  const drawingTools = (
    <div className="flex items-center gap-1 shrink-0 min-w-0">
      {addMenu}
      {SEPARATOR}
      <InkTools ink={ink} />
      {SEPARATOR}
      <ToolButton icon={CircleDashed} label="Select on canvas" active={tool === "select"} onClick={() => onToggleTool("select")} />
      {isErasing && <span className="text-[10px] text-ink-faint ml-1">(eraser)</span>}
      {tool === "select" && <span className="text-[10px] text-ink-faint ml-1">(draw a lasso to select)</span>}
      {selectedCount > 0 && <span className="text-[10px] text-ink-faint ml-1">({selectedCount} selected — drag, resize or Esc)</span>}
    </div>
  )

  const viewTools = (
    <div className="flex items-center gap-1 shrink-0">
      {tool === "screenshot" && <span className="text-[10px] text-ink-faint mr-1">(drag to capture)</span>}
      <ToolButton icon={Camera} label="Capture canvas" active={tool === "screenshot"} onClick={() => onToggleTool("screenshot")} />
      {SEPARATOR}
      <UndoRedoButtons onUndo={onUndo} onRedo={onRedo} canUndo={canUndo} canRedo={canRedo} separator={SEPARATOR} />
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
