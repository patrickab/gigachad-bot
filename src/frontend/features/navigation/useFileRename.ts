"use client"

import { useRenameDocument } from "@/contexts/ProjectContext"

/** The file a rename is about: `scope` is its project's slug, empty for a mounted file shown outside any project. */
export interface RenameTarget {
  path: string
  name: string
  scope: string
}

interface FileRenameOptions {
  /** The path `target` has now, when opening it moved the file since the row showed it. */
  currentPath: (target: RenameTarget) => string
  /** Called once the backend renamed `oldPath` to `newPath` and every reference followed. */
  onRenamed: (oldPath: string, newPath: string) => Promise<void> | void
}

/** Rename of sidebar files: project files (stored or on disk) and mounted files. Rejects with the reason, e.g. a taken name. */
export function useFileRename({ currentPath, onRenamed }: FileRenameOptions) {
  const renameDocument = useRenameDocument()
  const renameFile = async (target: RenameTarget, name: string) => {
    const path = currentPath(target)
    const renamed = await renameDocument(target.scope, path, name)
    if (renamed.path !== path) await onRenamed(path, renamed.path)
  }
  return { renameFile }
}
