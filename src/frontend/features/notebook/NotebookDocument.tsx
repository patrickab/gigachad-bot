"use client"

import dynamic from "next/dynamic"
import { useEffect, useState } from "react"
import { LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import type { CellTerminal } from "./cellTerminal"
import { onFileMove, type FileMove } from "@/lib/backend/fileEvents"

// The notebook libraries touch `document` while they load, so the editor is a client-only chunk.
const NotebookEditor = dynamic(() => import("./NotebookEditor").then((module) => module.NotebookEditor), { ssr: false, loading: () => <LoadingPlaceholder /> })

/** The path the open editor follows. A rename (a file move whose source is the shown path) moves it along and keeps the editor, any other path (another file opened in the tab) starts a fresh editor. */
function useEditorPath(path: string): { path: string; generation: number } {
  const [current, setCurrent] = useState({ path, generation: 0 })
  const [moves, setMoves] = useState<FileMove[]>([])
  useEffect(() => onFileMove((move) => setMoves((seen) => [...seen.slice(-19), move])), [])
  if (path !== current.path) {
    const renamed = moves.some((move) => move.from === current.path && move.to === path)
    setCurrent({ path, generation: renamed ? current.generation : current.generation + 1 })
  }
  return current
}

/** A notebook (`.ipynb`) of the file at `path`, shown by the tab `tabId`. Keyed by the tab, so a rename never remounts the editor. A `cellTerminal` edits the active cell in Neovim, without one the cells are CodeMirror. */
export function NotebookDocument({ path, tabId, cellTerminal = null }: { path: string; tabId: string; cellTerminal?: CellTerminal | null }) {
  const editor = useEditorPath(path)
  return <NotebookEditor key={`${tabId}:${editor.generation}`} path={editor.path} tabId={tabId} cellTerminal={cellTerminal} />
}
