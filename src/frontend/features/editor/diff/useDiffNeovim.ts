import { useCallback, useRef, useState } from "react"
import { useFollowFiles } from "@/hooks/useFollowFiles"
import { fetchRootDiffContent, loadFileViewerText, writeRootFile } from "@/lib/backend/api"
import { baseName } from "@/lib/paths"
import { errorMessage } from "@/lib/utils"
import { useNeovimTerminal, type NeovimDocument } from "@/features/editor/NeovimEditor"

/** Edit one changed file of `root` in Neovim against HEAD; `onReturn` runs when the editor closes. */
export function useDiffNeovim(root: string, onReturn: () => void) {
  const [editing, setEditing] = useState<NeovimDocument | null>(null)
  const [openingPath, setOpeningPath] = useState<string | null>(null)
  const [editError, setEditError] = useState("")
  // The file's text as last read or written, so the echo of this editor's own save is not pushed back
  const diskText = useRef<string | null>(null)
  const session = useNeovimTerminal(editing, async (text) => {
    if (!editing) throw new Error("No file selected")
    await writeRootFile(editing.path, text)
    diskText.current = text
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
      diskText.current = contents.content
      setEditing({ key: path, path, filename: baseName(path), ...contents })
    } catch (error) {
      setEditError(errorMessage(error, "Cannot edit this file"))
    } finally {
      setOpeningPath(null)
    }
  }, [root])
  const editFile = useCallback((path: string) => void openEditor(path), [openEditor])

  // A change of the edited file on disk, by anyone, is merged into the buffer
  useFollowFiles(editing ? [editing.path] : [], (path) => {
    void loadFileViewerText(path).then((text) => {
      if (text === diskText.current) return
      diskText.current = text
      session.setContent(text, path)
    }).catch(() => {})
  })

  return { editing, openingPath, editError, clearEditError: () => setEditError(""), editFile, session }
}
