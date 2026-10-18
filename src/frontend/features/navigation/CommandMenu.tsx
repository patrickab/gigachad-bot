"use client"

import { useEffect, useMemo, useState } from "react"
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
}

/** Filter commands inline and capture navigation keys while the search input keeps focus. */
export function CommandMenu({ open, items, query, onRun, onClose }: CommandMenuProps) {
  const [selectedIdx, setSelectedIdx] = useState(0)

  const filtered = useMemo(() => {
    if (!query) return items
    const q = query.toLowerCase()
    return items.filter((it) => it.command.toLowerCase().includes(q) || it.keywords?.some((k) => k.toLowerCase().includes(q)))
  }, [items, query])

  // Restart at the first entry whenever the filtered list changes.
  const [prevFiltered, setPrevFiltered] = useState(filtered)
  if (prevFiltered !== filtered) {
    setPrevFiltered(filtered)
    setSelectedIdx(0)
  }

  // Capture phase so the keys are handled before the focused input sees them.
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

  if (!open || filtered.length === 0) return null

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.12, ease: "easeOut" }}
          className="overflow-hidden"
        >
          <div className="mt-1 rounded-lg border border-divider/50 bg-paper py-1">
            {filtered.map((item, i) => (
              <div
                key={item.command}
                onMouseEnter={() => setSelectedIdx(i)}
                onClick={() => onRun(item.command)}
                className={cn(
                  "cursor-pointer px-2.5 py-1 font-mono text-[11px] text-ink transition-colors",
                  i === selectedIdx && "bg-surface-elevated",
                )}
              >
                {item.command}
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
