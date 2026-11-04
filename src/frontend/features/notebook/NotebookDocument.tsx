"use client"

import dynamic from "next/dynamic"
import { useEffect, useState } from "react"
import { LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import type { NotebookTextStore } from "./notebookTextStore"
import { onFileMove, type FileMove } from "@/lib/backend/fileEvents"

// The notebook libraries touch `document` while they load, so the editor is a client-only chunk.
const NotebookEditor = dynamic(() => import("./NotebookEditor").then((module) => module.NotebookEditor), { ssr: false, loading: () => <LoadingPlaceholder /> })

/** The file a notebook tab shows: `generation` counts the other files opened in the tab, a rename keeps it. */
export interface EditorPath {
  path: string
  generation: number
}

/** The path the open editor follows. A rename (a file move whose source is the shown path) moves it along and keeps the editor, any other path (another file opened in the tab) starts a fresh editor. */
export function useEditorPath(path: string): EditorPath {
  const [current, setCurrent] = useState({ path, generation: 0 })
  const [moves, setMoves] = useState<FileMove[]>([])
  useEffect(() => onFileMove((move) => setMoves((seen) => [...seen.slice(-19), move])), [])
  if (path !== current.path) {
    const renamed = moves.some((move) => move.from === current.path && move.to === path)
    setCurrent({ path, generation: renamed ? current.generation : current.generation + 1 })
  }
  return current
}

/**
 * A notebook (`.ipynb`) of the file `editor` (`useEditorPath`), shown by the tab `tabId`. Keyed by the tab and the generation, so a rename never remounts the editor and
 * another file gets a fresh one. Its model is bound into `textStore` while it is loaded.
 */
export function NotebookDocument({ editor, tabId, textStore }: { editor: EditorPath; tabId: string; textStore: NotebookTextStore }) {
  return <NotebookEditor key={`${tabId}:${editor.generation}`} path={editor.path} tabId={tabId} textStore={textStore} />
}
