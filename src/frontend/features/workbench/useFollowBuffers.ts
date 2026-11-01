"use client"

import { type RefObject } from "react"
import { useFollowFiles } from "@/hooks/useFollowFiles"
import { loadFileViewerText } from "@/lib/backend/api"

/**
 * Mirror every change of the open files from outside Neovim into it, whoever wrote it (the agent, a terminal, another device).
 *
 * - `synced` maps every file open in Neovim to the text Neovim and the disk last agreed on, the owner updates it on every save and every accepted push
 * - A change of a file in `paths` reads it and pushes it through `setContent` when it differs from its synced text, the backend merges it into unsaved edits
 * - A file closed or renamed while it was read is not pushed
 * - A failed read or a push Neovim could not take yet is left to the file's next change
 */
export function useFollowBuffers({ synced, paths, setContent }: {
  synced: RefObject<ReadonlyMap<string, string>>
  paths: readonly string[]
  setContent: (text: string, path: string) => boolean
}): void {
  const pull = async (path: string) => {
    let text: string
    try {
      text = await loadFileViewerText(path)
    } catch {
      // A missing file mid-write is normal, the write that completes it reports again
      return
    }
    if (!synced.current.has(path) || text === synced.current.get(path)) return
    setContent(text, path)
  }

  useFollowFiles(paths, (path) => { void pull(path) })
}
