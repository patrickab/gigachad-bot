"use client"

import { useEffect, useRef, type CSSProperties, type MouseEvent, type ReactNode, type Ref } from "react"
import { motion, useReducedMotion } from "framer-motion"
import { cn } from "@/lib/utils"
import { ModalBackdrop } from "@/components/ui/ModalBackdrop"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"

interface FloatingWindowProps {
  children: ReactNode
  onClose?: () => void
  /** Accessible name of the dialog. */
  label?: string
  overlayClassName?: string
  panelClassName?: string
}

/** Look of a floating window's panel, shared by `FloatingWindow` and the picker preview. */
export const FLOATING_PANEL_CLASS = "flex flex-col overflow-hidden rounded-3xl border border-divider/50 bg-paper shadow-[var(--shadow-xl)] outline-none"

/** Corner radius of `FLOATING_PANEL_CLASS` (rounded-3xl). */
export const FLOATING_PANEL_RADIUS_PX = 24

/** A floating window's panel: fades and grows in, instantly under reduced motion. */
export function FloatingPanel({ children, className, style, ref, onClick, ...aria }: {
  children?: ReactNode
  className?: string
  style?: CSSProperties
  ref?: Ref<HTMLDivElement>
  onClick?: (event: MouseEvent<HTMLDivElement>) => void
  role?: string
  tabIndex?: number
  "aria-modal"?: boolean
  "aria-label"?: string
  "aria-hidden"?: boolean
}) {
  const reduceMotion = useReducedMotion()
  return (
    <motion.div
      ref={ref}
      {...aria}
      initial={{ opacity: 0, scale: 0.98, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98, y: 3, transition: { duration: reduceMotion ? 0 : 0.12, ease: "easeIn" } }}
      transition={{ duration: reduceMotion ? 0 : 0.2, ease: "easeOut" }}
      onClick={onClick}
      style={style}
      className={cn(FLOATING_PANEL_CLASS, className)}
    >
      {children}
    </motion.div>
  )
}

/**
 * Modal overlay with a panel centered over the content area. Backdrop click and Escape call `onClose`.
 *
 * - Takes focus on open unless a child already focused itself, so Escape and keys reach it
 * - The panel fades and grows in, instantly under reduced motion
 */
export function FloatingWindow({ children, onClose, label, overlayClassName, panelClassName }: FloatingWindowProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  useEscapeLayer(() => onClose?.(), onClose !== undefined)
  useEffect(() => {
    const panel = panelRef.current
    if (panel && !panel.contains(document.activeElement)) panel.focus()
  }, [])

  return (
    <ModalBackdrop onClose={onClose} className={cn("z-[90] p-3 sm:p-6", overlayClassName)}>
      <FloatingPanel
        ref={panelRef}
        role="dialog"
        aria-modal
        aria-label={label}
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
        className={cn("h-[85vh] max-h-[800px] w-full max-w-6xl", panelClassName)}
      >
        {children}
      </FloatingPanel>
    </ModalBackdrop>
  )
}
