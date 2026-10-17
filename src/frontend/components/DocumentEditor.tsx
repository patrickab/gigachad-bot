"use client"

import { CanvasDocumentEditor } from "./CanvasDocumentEditor"
import { FileCanvasEditor } from "./FileCanvasEditor"
import { ArchitectureSourceEditor, TextDocumentEditor, type DocumentEditorProps } from "./TextDocumentEditor"
import { isArchitecturePath } from "@/lib/architecturePaths"

/** Route a document path to its editor: architecture source, canvas (file-root or collaborative), or text. */
export function DocumentEditor(props: DocumentEditorProps) {
  if (isArchitecturePath(props.path)) return <ArchitectureSourceEditor {...props} />
  if (props.path.endsWith(".canvas")) {
    const { persistOverride } = props
    return persistOverride ? <FileCanvasEditor {...props} persistOverride={persistOverride} /> : <CanvasDocumentEditor {...props} />
  }
  return <TextDocumentEditor {...props} />
}
