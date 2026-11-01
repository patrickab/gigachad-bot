"use client"

import { useEffect, useEffectEvent } from "react"
import { onFileChange } from "@/lib/backend/fileEvents"
import { watchFiles } from "@/lib/backend/fileWatch"
import { isInside } from "@/lib/paths"

/** Disk paths are absolute, a stored key like `project/notes.md` is not and gets its change events from the sync feed. */
const isDiskPath = (path: string) => path.startsWith("/")

/**
 * Call `onChange` with the path of every file in `files`, or inside `folders`, that changes, whoever wrote it.
 *
 * - Absolute paths are watched on disk, stored keys are not watched and hear the sync feed's change events
 * - A folder reports the changed file inside it, and once per reconnect its own path
 * - A change made by this device can report too, so compare what is read with what is shown
 * - The latest `onChange` is always called and the arrays may be fresh each render, only their content decides what is watched
 */
export function useFollowFiles(files: readonly string[], onChange: (path: string) => void, folders: readonly string[] = []): void {
  const changed = useEffectEvent(onChange)
  const filesKey = JSON.stringify([...new Set(files)].sort())
  const foldersKey = JSON.stringify([...new Set(folders)].sort())

  useEffect(() => {
    const followedFiles = new Set<string>(JSON.parse(filesKey))
    const followedFolders: string[] = JSON.parse(foldersKey)
    const stopWatching = watchFiles({ files: [...followedFiles].filter(isDiskPath), folders: followedFolders.filter(isDiskPath) })
    const stopListening = onFileChange(
      (path) => followedFiles.has(path) || followedFolders.some((folder) => isInside(path, folder)),
      (path) => changed(path),
    )
    return () => {
      stopListening()
      stopWatching()
    }
  }, [filesKey, foldersKey])
}
