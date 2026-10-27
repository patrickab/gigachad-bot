"use client"

import { useEffect, useRef, type CSSProperties, type ReactNode } from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"
import { ModalBackdrop } from "@/components/ui/ModalBackdrop"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"

interface FloatingWindowProps {
  children: ReactNode
  onClose?: () => void
  /** Accessible name of the dialog. */
  label?: string
  overlayClassName?: string
  /** Inline overlay style, e.g. padding that centers the panel over part of the screen. */
  overlayStyle?: CSSProperties
  panelClassName?: string
}

/**
 * Modal overlay with a centered panel. Backdrop click and Escape call `onClose`.
 *
 * - Takes focus on open unless a child already focused itself, so Escape and keys reach it
 */
export function FloatingWindow({ children, onClose, label, overlayClassName, overlayStyle, panelClassName }: FloatingWindowProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  useEscapeLayer(() => onClose?.(), onClose !== undefined)
  useEffect(() => {
    const panel = panelRef.current
    if (panel && !panel.contains(document.activeElement)) panel.focus()
  }, [])

  return (
    <ModalBackdrop onClose={onClose} className={cn("z-[90] p-3 sm:p-6", overlayClassName)} style={overlayStyle}>
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
    </ModalBackdrop>
  )
}
