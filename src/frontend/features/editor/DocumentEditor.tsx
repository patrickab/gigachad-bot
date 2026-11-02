"use client"

import { useContext, useId } from "react"
import { CanvasDocumentEditor } from "@/features/editor/canvas/CanvasDocumentEditor"
import { FileCanvasEditor } from "@/features/editor/canvas/FileCanvasEditor"
import { ArchitectureSourceEditor, TextDocumentEditor, type DocumentEditorProps } from "./TextDocumentEditor"
import { RawFileView } from "./RawFileView"
import { EditorShell } from "./EditorShell"
import { NotebookDocument } from "@/features/notebook/NotebookDocument"
import { TabContext } from "@/contexts/TabContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"
import { isNotebookPath, kindOf } from "@/lib/paths"

/** An image file as itself, from its raw URL: read-only, shrunk to fit the pane and never enlarged. */
function ImageDocument({ path, overlay, onClose }: Pick<DocumentEditorProps, "path" | "overlay" | "onClose">) {
  useEscapeLayer(onClose, !!overlay)
  return <div className="flex h-full flex-col bg-paper"><RawFileView path={path} kind="image" /></div>
}

/** Route a document path to its viewer or editor: image, notebook, architecture source, canvas (file-root or collaborative), or text. */
export function DocumentEditor(props: DocumentEditorProps) {
  const tabId = useContext(TabContext)?.tab.id
  const ownId = useId()
  if (kindOf(props.path) === "image") return <ImageDocument path={props.path} overlay={props.overlay} onClose={props.onClose} />
  if (isNotebookPath(props.path)) return <EditorShell overlay={!!props.overlay} fullscreen={false}><NotebookDocument path={props.path} tabId={tabId ?? ownId} /></EditorShell>
  if (isArchitecturePath(props.path)) return <ArchitectureSourceEditor {...props} />
  if (props.path.endsWith(".canvas")) {
    const { persistOverride } = props
    return persistOverride ? <FileCanvasEditor {...props} persistOverride={persistOverride} /> : <CanvasDocumentEditor {...props} />
  }
  return <TextDocumentEditor {...props} />
}
