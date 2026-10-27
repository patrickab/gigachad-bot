"use client"

import type { CSSProperties, ReactNode } from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

/** Full-screen fading overlay that centers its children and calls `onClose` when the backdrop itself is clicked. */
export function ModalBackdrop({ onClose, className, style, children }: { onClose?: () => void; className?: string; style?: CSSProperties; children: ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      className={cn("fixed inset-0 flex items-center justify-center bg-backdrop backdrop-blur-[2px]", className)}
      onClick={onClose}
      style={style}
    >
      {children}
    </motion.div>
  )
}
