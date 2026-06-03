"use client"

import { ElementType, ReactNode, useCallback, useEffect, useRef, useState } from "react"
import { motion } from "framer-motion"
import { PanelRightClose, PanelRightOpen } from "lucide-react"
import { ExpandableSidebarElement } from "./ExpandableSidebarElement"
import { CHROME_UNIT_PX } from "@/lib/config"

const COLLAPSED_WIDTH = 50
const DEFAULT_EXPANDED_WIDTH = 320
const MIN_EXPANDED_WIDTH = 200
const MAX_EXPANDED_WIDTH = 600

export interface ChatSidebarElementConfig {
  id: string
  icon: ElementType
  title: string
  badge?: ReactNode
  body: ReactNode
}

interface ChatSidebarProps {
  collapsed: boolean
  onToggle: () => void
  elements: ChatSidebarElementConfig[]
}

export function ChatSidebar({ collapsed, onToggle, elements }: ChatSidebarProps) {
  const [width, setWidth] = useState(DEFAULT_EXPANDED_WIDTH)
  const [mounted, setMounted] = useState(false)
  const dragging = useRef(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      dragging.current = true
      const startX = e.clientX
      const startWidth = width

      const onMouseMove = (ev: MouseEvent) => {
        if (!dragging.current) return
        const newPx = startWidth + (startX - ev.clientX)
        setWidth(Math.min(MAX_EXPANDED_WIDTH, Math.max(MIN_EXPANDED_WIDTH, newPx)))
      }

      const onMouseUp = () => {
        dragging.current = false
        document.removeEventListener("mousemove", onMouseMove)
        document.removeEventListener("mouseup", onMouseUp)
      }

      document.addEventListener("mousemove", onMouseMove)
      document.addEventListener("mouseup", onMouseUp)
    },
    [width]
  )

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? COLLAPSED_WIDTH : width }}
      transition={{ duration: 0.25, ease: "easeInOut" }}
      className={`shrink-0 border-l border-zinc-800 bg-zinc-950 flex flex-col overflow-hidden relative ${mounted ? "" : "!transition-none"}`}
    >
      {!collapsed && (
        <div
          className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-zinc-700 active:bg-blue-500/50 transition-colors z-10"
          onMouseDown={handleMouseDown}
        />
      )}
      <div
        className="flex items-center gap-2 px-3 border-b border-zinc-800/50"
        style={{ height: `${CHROME_UNIT_PX}px` }}
      >
        {collapsed ? (
          <button
            onClick={onToggle}
            className="w-full flex items-center justify-center rounded p-1.5 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 transition-colors"
            title="Expand sidebar"
          >
            <PanelRightOpen className="h-4 w-4" />
          </button>
        ) : (
          <>
            <span className="text-sm font-semibold text-zinc-200 px-1">Chat</span>
            <div className="flex-1" />
            <button
              onClick={onToggle}
              className="rounded p-0.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
              title="Collapse sidebar"
            >
              <PanelRightClose className="h-3.5 w-3.5" />
            </button>
          </>
        )}
      </div>
      {collapsed ? (
        <div className="flex-1 overflow-y-auto py-2 flex flex-col items-center gap-1">
          {elements.map(({ id, icon: Icon, title }) => (
            <button
              key={id}
              className="flex h-8 w-8 items-center justify-center rounded-md text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200 transition-colors"
              title={title}
            >
              <Icon className="h-4 w-4" />
            </button>
          ))}
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          {elements.map((el) => (
            <ExpandableSidebarElement
              key={el.id}
              id={el.id}
              icon={el.icon}
              title={el.title}
              badge={el.badge}
            >
              {el.body}
            </ExpandableSidebarElement>
          ))}
        </div>
      )}
    </motion.aside>
  )
}
