"use client"

import { useEffect, type ElementType, type ReactNode, type RefObject } from "react"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { ModalBackdrop } from "@/components/ui/ModalBackdrop"
import { motion, AnimatePresence, useReducedMotion } from "framer-motion"

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
  /** Max-width class of the card, `max-w-sm` by default. */
  widthClassName?: string
  children: ReactNode
}

/** Centered modal shell: backdrop, animated card with icon + title header, Escape and backdrop-click close. The card fades and grows in, instantly under reduced motion. */
export function Dialog({ open, onClose, icon: Icon, title, initialFocusRef, panelClassName = DEFAULT_PANEL_CLASS, widthClassName = "max-w-sm", children }: DialogProps) {
  const reduceMotion = useReducedMotion()
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
        <ModalBackdrop onClose={onClose} className="z-[100]">
          <motion.div
            initial={{ opacity: 0, scale: 0.97, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98, y: 2, transition: { duration: reduceMotion ? 0 : 0.12, ease: "easeIn" } }}
            transition={{ duration: reduceMotion ? 0 : 0.18, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
            className={`w-full ${widthClassName} mx-4 overflow-hidden ${panelClassName}`}
          >
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-center gap-2 text-sm font-medium text-ink">
                <Icon className="h-4 w-4 text-ink" />
                {title}
              </div>
            </div>
            {children}
          </motion.div>
        </ModalBackdrop>
      )}
    </AnimatePresence>
  )
}
