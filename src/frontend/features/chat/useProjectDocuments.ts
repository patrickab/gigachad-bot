"use client"

import { useCallback, useEffect, useMemo, useState, type Dispatch, type RefObject, type SetStateAction } from "react"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import type { AppSurface } from "@/contexts/TabContext"
import {
  addDocument,
  attachReference,
  listAllDocuments,
  listProjectDocuments,
  loadFileViewerText,
  uploadFile,
} from "@/lib/backend/api"
import { architectureSource, isArchitecturePath } from "@/lib/domain/architecturePaths"
import { renderCanvasTextToJpeg } from "@/lib/domain/canvasRender"
import type { ProjectDocument } from "@/lib/domain/types"
import { baseName } from "@/lib/paths"
import { errorMessage } from "@/lib/utils"
import { useEventListener } from "@/hooks/useEventListener"

/** Own the document picker: project and global lists, selection into chat attachments, and the Alt+X shortcut. */
export function useProjectDocuments({
  isActive,
  appMode,
  projectSlug,
  chatId,
  chatInputRef,
  setExtracting,
}: {
  isActive: boolean
  appMode: AppSurface
  projectSlug: string | null
  chatId: string
  chatInputRef: RefObject<ChatInputHandle | null>
  setExtracting: Dispatch<SetStateAction<number>>
}) {
  const [documentOpen, setDocumentOpen] = useState(false)
  const [projectDocuments, setProjectDocuments] = useState<ProjectDocument[]>([])
  const [architectureError, setArchitectureError] = useState("")
  const [allDocuments, setAllDocuments] = useState<ProjectDocument[]>([])

  const architectureSources = useMemo(() => projectDocuments.filter((document) => isArchitecturePath(document.path)), [projectDocuments])

  const handleDocumentSelect = useCallback(async (path: string) => {
    setDocumentOpen(false)
    // Canvas docs are attached as a rendered JPEG of their saved content
    if (path.endsWith(".canvas")) {
      try {
        const blob = await renderCanvasTextToJpeg(await loadFileViewerText(path))
        if (!blob) return
        const name = baseName(path).replace(/\.canvas$/, ".jpg")
        const file = new File([blob], name, { type: "image/jpeg" })
        const att = await uploadFile(chatId, file, projectSlug, true)
        chatInputRef.current?.addAttachment(att)
      } catch { /* */ }
    } else {
      setExtracting((n) => n + 1)
      try { chatInputRef.current?.addAttachment(await attachReference(path)) } catch { /* */ }
      finally { setExtracting((n) => n - 1) }
    }
  }, [chatInputRef, setExtracting, chatId, projectSlug])

  const loadProjectDocs = useCallback(() => {
    if (projectSlug) listProjectDocuments(projectSlug).then(setProjectDocuments).catch(() => {})
  }, [projectSlug])

  const refreshDocuments = useCallback(() => {
    loadProjectDocs()
    listAllDocuments().then(setAllDocuments).catch(() => {})
  }, [loadProjectDocs])

  useEffect(() => { loadProjectDocs() }, [loadProjectDocs])

  // Without a project there are no documents, whatever was loaded for the previous one
  if (!projectSlug && projectDocuments.length > 0) setProjectDocuments([])

  // One text attachment holding every `.c4` source as a fenced LikeC4 block under its workspace path.
  const handleAttachArchitecture = useCallback(async () => {
    if (!projectSlug) return
    setArchitectureError("")
    setExtracting((count) => count + 1)
    try {
      const texts = await Promise.all(architectureSources.map(async (source) => ({ path: architectureSource(source.path), text: await loadFileViewerText(source.path) })))
      const content = `# ${projectSlug} architecture\n\n${texts.map(({ path, text }) => `## ${path}\n\n\`\`\`likec4\n${text}\n\`\`\``).join("\n\n")}`
      const file = new File([content], `${projectSlug}-architecture-${crypto.randomUUID().slice(0, 8)}.txt`, { type: "text/plain" })
      const attachment = await uploadFile(chatId, file, projectSlug)
      chatInputRef.current?.addAttachment({ ...attachment, content })
    } catch (cause) {
      setArchitectureError(errorMessage(cause, "Could not attach architecture"))
    } finally {
      setExtracting((count) => count - 1)
    }
  }, [projectSlug, architectureSources, chatId, chatInputRef, setExtracting])

  const openDocuments = useCallback(() => {
    refreshDocuments()
    setDocumentOpen(true)
  }, [refreshDocuments])

  const handleAddDocToProject = useCallback(async (path: string) => {
    if (!projectSlug) return
    try {
      await addDocument(projectSlug, path)
      refreshDocuments()
    } catch { }
  }, [projectSlug, refreshDocuments])

  useEventListener("window", "keydown", (e) => {
    // Alt+X toggles the document picker
    if (e.altKey && (e.key === "x" || e.key === "X")) {
      e.preventDefault()
      setDocumentOpen((open) => {
        if (!open) refreshDocuments()
        return !open
      })
    }
  }, { active: isActive && appMode === "chat" })

  return {
    documentOpen,
    setDocumentOpen,
    projectDocuments,
    allDocuments,
    architectureError,
    /** Absent while the project has no architecture sources to attach. */
    handleAttachArchitecture: architectureSources.length > 0 ? handleAttachArchitecture : undefined,
    refreshDocuments,
    openDocuments,
    handleDocumentSelect,
    handleAddDocToProject,
  }
}
