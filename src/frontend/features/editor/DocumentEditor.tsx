"use client"

import { CanvasDocumentEditor } from "@/features/editor/canvas/CanvasDocumentEditor"
import { FileCanvasEditor } from "@/features/editor/canvas/FileCanvasEditor"
import { ArchitectureSourceEditor, TextDocumentEditor, type DocumentEditorProps } from "./TextDocumentEditor"
import { RawFileView } from "./RawFileView"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"
import { kindOf } from "@/lib/paths"

/** An image file as itself, from its raw URL: read-only, shrunk to fit the pane and never enlarged. */
function ImageDocument({ path, overlay, onClose }: Pick<DocumentEditorProps, "path" | "overlay" | "onClose">) {
  useEscapeLayer(onClose, !!overlay)
  return <div className="flex h-full flex-col bg-paper"><RawFileView path={path} kind="image" /></div>
}

/** Route a document path to its viewer or editor: image, architecture source, canvas (file-root or collaborative), or text. Notebooks open in the tab's `NotebookWorkbench`. */
export function DocumentEditor(props: DocumentEditorProps) {
  if (kindOf(props.path) === "image") return <ImageDocument path={props.path} overlay={props.overlay} onClose={props.onClose} />
  if (isArchitecturePath(props.path)) return <ArchitectureSourceEditor {...props} />
  if (props.path.endsWith(".canvas")) {
    const { persistOverride } = props
    return persistOverride ? <FileCanvasEditor {...props} persistOverride={persistOverride} /> : <CanvasDocumentEditor {...props} />
  }
  return <TextDocumentEditor {...props} />
}
