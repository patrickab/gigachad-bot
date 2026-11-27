import type { ReactNode } from "react"
import { motion } from "framer-motion"

/** Frame of a document editor: a full-pane overlay. */
export function EditorShell({ children }: { children: ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.15 }}
      className="absolute inset-0 z-30 flex flex-row bg-paper"
    >
      <div className="flex-1 min-w-0 flex flex-col relative">
        {children}
      </div>
    </motion.div>
  )
}
