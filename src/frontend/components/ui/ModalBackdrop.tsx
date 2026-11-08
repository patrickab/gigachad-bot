"use client"

import { useState, type ReactNode } from "react"
import { motion, useReducedMotion } from "framer-motion"
import { cn } from "@/lib/utils"
import { useContentInsets } from "@/components/ui/ContentArea"

/** Full-screen fading overlay that centers its children over the content area and calls `onClose` when the backdrop itself is clicked. Fades instantly under reduced motion. */
export function ModalBackdrop({ onClose, className, children }: { onClose?: () => void; className?: string; children?: ReactNode }) {
  const [backdrop, setBackdrop] = useState<HTMLDivElement | null>(null)
  const insets = useContentInsets(backdrop)
  const reduceMotion = useReducedMotion()
  return (
    <motion.div
      ref={setBackdrop}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: reduceMotion ? 0 : 0.12, ease: "easeIn" } }}
      transition={{ duration: reduceMotion ? 0 : 0.18, ease: "easeOut" }}
      className={cn("fixed inset-0 flex items-center justify-center bg-backdrop backdrop-blur-[2px]", className)}
      onClick={onClose}
    >
      <div className="flex h-full min-w-0 flex-1 items-center justify-center" style={{ marginLeft: insets?.left, marginRight: insets?.right }}>
        {children}
      </div>
    </motion.div>
  )
}
