"use client"

import { useState, useRef, useEffect, type ElementType } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Save } from "lucide-react"
import { cn } from "@/lib/utils"

interface SaveChatModalProps {
  open: boolean
  onClose: () => void
  onSave: (name: string) => void
  title?: string
  icon?: ElementType
}

export function SaveChatModal({ open, onClose, onSave, title = "Save Chat", icon: Icon = Save }: SaveChatModalProps) {
  const [name, setName] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setName("")
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [open, onClose])

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-backdrop backdrop-blur-[2px]"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-sm mx-4 rounded-xl border border-divider bg-paper shadow-[var(--shadow-xl)] overflow-hidden"
          >
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-center gap-2 text-sm font-medium text-ink">
                <Icon className="h-4 w-4 text-ink" />
                {title}
              </div>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault()
                if (name.trim()) onSave(name.trim())
              }}
              className="px-5 pb-5"
            >
              <input
                ref={inputRef}
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className={cn(
                  "w-full rounded-lg border border-divider bg-surface/60 px-3 py-2.5 text-sm text-ink placeholder-ink-faint",
                  "outline-none transition-all duration-200",
                  "focus:border-ink-muted"
                )}
                autoComplete="off"
                spellCheck={false}
              />
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
