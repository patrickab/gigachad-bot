"use client"

import { useEffect, type ElementType, type ReactNode, type RefObject } from "react"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { motion, AnimatePresence } from "framer-motion"

const DEFAULT_PANEL_CLASS = "rounded-xl border border-divider bg-paper shadow-[var(--shadow-xl)]"

interface DialogProps {
  open: boolean
  onClose: () => void
  icon: ElementType
  title: ReactNode
  /** Element focused once the enter animation has mounted it. */
  initialFocusRef?: RefObject<HTMLElement | null>
  /** Surface classes of the card; sizing and clipping are fixed by the shell. */
  panelClassName?: string
  children: ReactNode
}

/** Centered modal shell: backdrop, animated card with icon + title header, Escape and backdrop-click close. */
export function Dialog({ open, onClose, icon: Icon, title, initialFocusRef, panelClassName = DEFAULT_PANEL_CLASS, children }: DialogProps) {
  useEffect(() => {
    if (!open) return
    // Wait for the enter animation to mount the input before focusing
    const timer = setTimeout(() => initialFocusRef?.current?.focus(), 50)
    return () => clearTimeout(timer)
  }, [open, initialFocusRef])

  useEscapeLayer(onClose, open)

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
            className={`w-full max-w-sm mx-4 overflow-hidden ${panelClassName}`}
          >
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-center gap-2 text-sm font-medium text-ink">
                <Icon className="h-4 w-4 text-ink" />
                {title}
              </div>
            </div>
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
