"use client"

import { CanvasDocumentEditor } from "@/features/editor/canvas/CanvasDocumentEditor"
import { FileCanvasEditor } from "@/features/editor/canvas/FileCanvasEditor"
import { ArchitectureSourceEditor, TextDocumentEditor, type DocumentEditorProps } from "./TextDocumentEditor"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"

/** Route a document path to its editor: architecture source, canvas (file-root or collaborative), or text. */
export function DocumentEditor(props: DocumentEditorProps) {
  if (isArchitecturePath(props.path)) return <ArchitectureSourceEditor {...props} />
  if (props.path.endsWith(".canvas")) {
    const { persistOverride } = props
    return persistOverride ? <FileCanvasEditor {...props} persistOverride={persistOverride} /> : <CanvasDocumentEditor {...props} />
  }
  return <TextDocumentEditor {...props} />
}
