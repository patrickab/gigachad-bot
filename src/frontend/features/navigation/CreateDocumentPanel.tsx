"use client"

import { useState } from "react"
import { FilePlus } from "lucide-react"
import { PillTabs } from "@/components/ui/PillTabs"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { useResetOnOpen } from "@/hooks/useResetOnChange"

type DocType = "md" | "tex" | "canvas" | "c4"

const DOC_TYPES: readonly { value: DocType; label: string }[] = [
  { value: "md", label: "Markdown" },
  { value: "tex", label: "LaTeX" },
  { value: "canvas", label: "Drawing" },
  { value: "c4", label: "Architecture" },
]

interface CreateDocumentPanelProps {
  open: boolean
  onClose: () => void
  onCreate: (name: string) => void
}

/** Modal that picks a document type and name, passing the full filename to `onCreate`. */
export function CreateDocumentPanel({ open, onClose, onCreate }: CreateDocumentPanelProps) {
  const [type, setType] = useState<DocType>("md")

  // Reset to the default type each time the modal opens
  useResetOnOpen(open, () => setType("md"))

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
          <PillTabs options={DOC_TYPES} value={type} onChange={setType} />
        </div>
      }
    />
  )
}
