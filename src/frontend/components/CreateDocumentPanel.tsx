"use client"

import { useState, useEffect } from "react"
import { FilePlus } from "lucide-react"
import { cn } from "@/lib/utils"
import { SaveAsModal } from "./SaveAsModal"

type DocType = "md" | "tex" | "canvas" | "c4"

const DOC_TYPE_LABELS: Record<DocType, string> = { md: "Markdown", tex: "LaTeX", canvas: "Drawing", c4: "Architecture" }

interface CreateDocumentPanelProps {
  open: boolean
  onClose: () => void
  onCreate: (name: string) => void
}

/** Modal that picks a document type and name, passing the full filename to `onCreate`. */
export function CreateDocumentPanel({ open, onClose, onCreate }: CreateDocumentPanelProps) {
  const [type, setType] = useState<DocType>("md")

  useEffect(() => {
    if (open) setType("md")
  }, [open])

  const ext = `.${type}`

  return (
    <SaveAsModal
      open={open}
      onClose={onClose}
      // Append the type's extension unless the user already typed it
      onSave={(name) => onCreate(name.endsWith(ext) ? name : `${name}${ext}`)}
      title="New Document"
      icon={FilePlus}
      placeholder={type === "c4" ? "backend/backend.c4" : `filename${ext}`}
      beforeInput={
        <div className="px-5 pb-2">
          <div className="flex gap-1">
            {(Object.keys(DOC_TYPE_LABELS) as DocType[]).map((t) => (
              <button
                key={t}
                onClick={() => setType(t)}
                className={cn(
                  "rounded-full px-3 py-1 text-[11px] font-medium transition-colors",
                  type === t
                    ? "bg-surface-elevated text-ink"
                    : "text-ink-muted hover:text-ink hover:bg-surface/50"
                )}
              >
                {DOC_TYPE_LABELS[t]}
              </button>
            ))}
          </div>
        </div>
      }
    />
  )
}
