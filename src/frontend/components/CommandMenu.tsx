"use client"

import { useEffect, useMemo, useState } from "react"
import { createPortal } from "react-dom"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"

export interface CommandMenuItem {
  command: string
  keywords?: string[]
}

interface CommandMenuProps {
  open: boolean
  items: CommandMenuItem[]
  query: string
  onRun: (command: string) => void
  onClose: () => void
  container: HTMLElement | null
}

export function CommandMenu({ open, items, query, onRun, onClose, container }: CommandMenuProps) {
  const [selectedIdx, setSelectedIdx] = useState(0)

  const filtered = useMemo(() => {
    if (!query) return items
    const q = query.toLowerCase()
    return items.filter((it) => it.command.toLowerCase().includes(q) || it.keywords?.some((k) => k.toLowerCase().includes(q)))
  }, [items, query])

  useEffect(() => { setSelectedIdx(0) }, [filtered])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onClose(); return }
      if (e.key === "ArrowDown") { e.preventDefault(); setSelectedIdx((i) => (i + 1) % filtered.length); return }
      if (e.key === "ArrowUp") { e.preventDefault(); setSelectedIdx((i) => (i - 1 + filtered.length) % filtered.length); return }
      if (e.key === "Enter" && filtered.length > 0) { e.preventDefault(); onRun(filtered[selectedIdx]?.command ?? filtered[0].command) }
    }
    window.addEventListener("keydown", onKey, true)
    return () => window.removeEventListener("keydown", onKey, true)
  }, [open, onClose, onRun, filtered, selectedIdx])

  const dropdownWidth = useMemo(() => {
    if (!open || !container) return undefined
    const tabBar = document.querySelector<HTMLElement>("[data-tabbar]")
    if (!tabBar) return undefined
    const barW = tabBar.clientWidth
    const tabCount = tabBar.children.length
    const tabW = tabCount > 0 ? barW / tabCount : barW
    if (tabW < barW * 0.25) return `${tabW}px`
    const minW = barW * 0.2
    const maxW = tabW * 0.33
    return `${Math.max(minW, maxW)}px`
  }, [open, container])

  if (!container || filtered.length === 0) return null

  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0, y: -2 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -2 }}
          transition={{ duration: 0.12, ease: "easeOut" }}
          style={{ width: dropdownWidth }}
          className="absolute left-0 z-[100] origin-top-left overflow-hidden border-x border-b border-divider bg-paper shadow-[var(--shadow-lg)]"
        >
          {filtered.map((item, i) => (
            <div
              key={item.command}
              onMouseEnter={() => setSelectedIdx(i)}
              onClick={() => onRun(item.command)}
              className={cn(
                "cursor-pointer px-3 py-1 font-mono text-xs text-ink transition-colors",
                i === selectedIdx && "bg-surface-elevated",
              )}
            >
              {item.command}
            </div>
          ))}
        </motion.div>
      )}
    </AnimatePresence>,
    container,
  )
}
