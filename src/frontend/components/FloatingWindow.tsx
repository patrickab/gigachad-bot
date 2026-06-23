"use client"

import { useEffect, type ReactNode } from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface FloatingWindowProps {
  children: ReactNode
  onClose?: () => void
  overlayClassName?: string
  panelClassName?: string
}

export function FloatingWindow({ children, onClose, overlayClassName, panelClassName }: FloatingWindowProps) {
  useEffect(() => {
    if (!onClose) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopImmediatePropagation(); onClose() }
    }
    window.addEventListener("keydown", handler, true)
    return () => window.removeEventListener("keydown", handler, true)
  }, [onClose])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      className={cn("fixed inset-0 z-[90] flex items-center justify-center bg-backdrop p-3 backdrop-blur-[2px] sm:p-6", overlayClassName)}
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.98, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.98, y: 10 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
        onClick={(e) => e.stopPropagation()}
        className={cn(
          "flex h-[85vh] max-h-[800px] w-full max-w-6xl flex-col overflow-hidden rounded-3xl border border-divider/50 bg-paper shadow-[var(--shadow-xl)]",
          panelClassName,
        )}
      >
        {children}
      </motion.div>
    </motion.div>
  )
}
