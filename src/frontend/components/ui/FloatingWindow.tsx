"use client"

import { useEffect, useRef, type ReactNode } from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"

interface FloatingWindowProps {
  children: ReactNode
  onClose?: () => void
  /** Accessible name of the dialog. */
  label?: string
  overlayClassName?: string
  panelClassName?: string
  /** Whether a click outside the panel calls `onClose`, false where that would lose unsaved work. */
  dismissOnBackdrop?: boolean
}

/**
 * Modal overlay with a centered panel. Backdrop click (unless `dismissOnBackdrop` is false) and Escape call `onClose`.
 *
 * - Takes focus on open unless a child already focused itself, so Escape and keys reach it even when opened from the omp frame
 */
export function FloatingWindow({ children, onClose, label, overlayClassName, panelClassName, dismissOnBackdrop = true }: FloatingWindowProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  useEscapeLayer(() => onClose?.(), onClose !== undefined)
  useEffect(() => {
    const panel = panelRef.current
    if (panel && !panel.contains(document.activeElement)) panel.focus()
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      className={cn("fixed inset-0 z-[90] flex items-center justify-center bg-backdrop p-3 backdrop-blur-[2px] sm:p-6", overlayClassName)}
      onClick={dismissOnBackdrop ? onClose : undefined}
    >
      <motion.div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={label}
        tabIndex={-1}
        initial={{ opacity: 0, scale: 0.98, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.98, y: 10 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
        onClick={(e) => e.stopPropagation()}
        className={cn(
          "flex h-[85vh] max-h-[800px] w-full max-w-6xl flex-col overflow-hidden rounded-3xl border border-divider/50 bg-paper shadow-[var(--shadow-xl)] outline-none",
          panelClassName,
        )}
      >
        {children}
      </motion.div>
    </motion.div>
  )
}
