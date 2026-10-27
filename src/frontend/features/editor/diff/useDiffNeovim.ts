import { useCallback, useState } from "react"
import { fetchRootDiffContent, writeRootFile } from "@/lib/backend/api"
import { baseName } from "@/lib/paths"
import { errorMessage } from "@/lib/utils"
import { useNeovimTerminal, type NeovimDocument } from "@/features/editor/NeovimEditor"

/** Edit one changed file of `root` in Neovim against HEAD; `onReturn` runs when the editor closes. */
export function useDiffNeovim(root: string, onReturn: () => void) {
  const [editing, setEditing] = useState<NeovimDocument | null>(null)
  const [openingPath, setOpeningPath] = useState<string | null>(null)
  const [editError, setEditError] = useState("")
  const session = useNeovimTerminal(editing, async (text) => {
    if (!editing) throw new Error("No file selected")
    await writeRootFile(editing.path, text)
    setEditing((current) => current?.key === editing.key ? { ...current, content: text } : current)
    return text
  }, () => {
    setEditing(null)
    onReturn()
  })

  const openEditor = useCallback(async (filePath: string) => {
    setOpeningPath(filePath)
    setEditError("")
    try {
      const contents = await fetchRootDiffContent(root, filePath)
      const path = `${root}/${filePath}`
      setEditing({ key: path, path, filename: baseName(path), ...contents })
    } catch (error) {
      setEditError(errorMessage(error, "Cannot edit this file"))
    } finally {
      setOpeningPath(null)
    }
  }, [root])
  const editFile = useCallback((path: string) => void openEditor(path), [openEditor])

  return { editing, openingPath, editError, clearEditError: () => setEditError(""), editFile, session }
}
