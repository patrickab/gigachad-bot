import type { ReactNode } from "react"
import { createPortal } from "react-dom"
import { motion, AnimatePresence } from "framer-motion"
import { ResizableEditor } from "./ResizableEditor"

/** Frame of a document editor: a full-pane overlay, a fullscreen portal, or a resizable block in the sidebar. */
export function EditorShell({ overlay, fullscreen, children }: { overlay: boolean; fullscreen: boolean; children: ReactNode }) {
  if (overlay) {
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

  if (fullscreen) {
    return createPortal(
      <AnimatePresence>
        <motion.div
          key="doc-editor-fullscreen"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-[100] flex flex-col bg-paper"
        >
          {children}
        </motion.div>
      </AnimatePresence>,
      document.body,
    )
  }

  return <ResizableEditor>{children}</ResizableEditor>
}
