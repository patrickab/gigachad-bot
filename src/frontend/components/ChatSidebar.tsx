"use client"

import { ElementType, ReactNode, useCallback, useEffect, useRef } from "react"
import { ChevronDown, ChevronRight } from "lucide-react"
import { motion, useMotionValue, useTransform, animate } from "framer-motion"
import { cn } from "@/lib/utils"

export const MIN_SIDEBAR_WIDTH = 200
export const MAX_SIDEBAR_WIDTH = 600

export interface ChatSidebarElementConfig {
  id: string
  icon: React.ElementType
  title: string
  badge?: React.ReactNode
  action?: React.ReactNode
  open: boolean
  onOpenChange: (open: boolean) => void
  body: React.ReactNode
}

function Collapse({ open, children }: { open: boolean; children: ReactNode }) {
  return (
    <div
      className={cn(
        "grid transition-[grid-template-rows] duration-200 ease-in-out",
        open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
      )}
    >
      <div className="overflow-hidden">{children}</div>
    </div>
  )
}

function ExpandableSidebarElement({ icon: Icon, title, badge, action, open, onOpenChange, children }: { icon: ElementType; title: string; badge?: ReactNode; action?: ReactNode; open: boolean; onOpenChange: (open: boolean) => void; children: ReactNode }) {
  return (
    <div className="border-b border-divider/50">
      <div className="flex items-center">
        <button
          onClick={() => onOpenChange(!open)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2 text-ink hover:bg-surface/50 transition-colors rounded"
        >
          {open ? (
            <ChevronDown className="h-3 w-3 text-ink-subtle shrink-0" />
          ) : (
            <ChevronRight className="h-3 w-3 text-ink-subtle shrink-0" />
          )}
          <Icon className="h-3.5 w-3.5 text-ink-muted shrink-0" />
          <span className="text-[11px] font-medium truncate flex-1 text-left">{title}</span>
          {badge != null && <span className="text-[10px] text-ink-faint shrink-0">{badge}</span>}
        </button>
        {action != null && <div className="shrink-0 pr-1.5">{action}</div>}
      </div>
      <Collapse open={open}>{children}</Collapse>
    </div>
  )
}

interface ChatSidebarProps {
  elements: ChatSidebarElementConfig[]
  width: number
  onWidthChange: (w: number) => void
  maxWidth?: number
}

export function ChatSidebar({ elements, width, onWidthChange, maxWidth: maxWidthProp }: ChatSidebarProps) {
  const effectiveMax = maxWidthProp ?? MAX_SIDEBAR_WIDTH
  const dragging = useRef(false)
  const motionWidth = useMotionValue(width)

  useEffect(() => {
    if (dragging.current) return
    const controls = animate(motionWidth, width, {
      type: "spring",
      stiffness: 400,
      damping: 40,
      mass: 0.6,
    })
    return controls.stop
  }, [width, motionWidth])

  const widthStyle = useTransform(motionWidth, (v) => `${v}px`)

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      dragging.current = true
      const startX = e.clientX
      const startWidth = motionWidth.get()

      const onMouseMove = (ev: MouseEvent) => {
        if (!dragging.current) return
        const newPx = startWidth + (startX - ev.clientX)
        const clamped = Math.min(effectiveMax, Math.max(MIN_SIDEBAR_WIDTH, newPx))
        motionWidth.set(clamped)
      }

      const onMouseUp = () => {
        if (!dragging.current) return
        dragging.current = false
        const finalWidth = motionWidth.get()
        const clamped = Math.min(effectiveMax, Math.max(MIN_SIDEBAR_WIDTH, finalWidth))
        motionWidth.set(clamped)
        onWidthChange(clamped)
        document.removeEventListener("mousemove", onMouseMove)
        document.removeEventListener("mouseup", onMouseUp)
      }

      document.addEventListener("mousemove", onMouseMove)
      document.addEventListener("mouseup", onMouseUp)
    },
    [motionWidth, onWidthChange, effectiveMax]
  )

  return (
    <motion.aside
      initial={false}
      style={{ width: widthStyle }}
      className="shrink-0 border-l border-divider bg-paper flex flex-col overflow-hidden relative"
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize sidebar"
        className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-surface-elevated active:bg-surface-elevated focus-visible:bg-surface-elevated transition-colors z-10"
        onMouseDown={handleMouseDown}
      />
      <div className="flex-1 overflow-y-auto">
        {elements.map((el) => (
          <ExpandableSidebarElement
            key={el.id}
            icon={el.icon}
            title={el.title}
            badge={el.badge}
            action={el.action}
            open={el.open}
            onOpenChange={el.onOpenChange}
          >
            {el.body}
          </ExpandableSidebarElement>
        ))}
      </div>
    </motion.aside>
  )
}
