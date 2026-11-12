"use client"

import { useCallback, useMemo, useState, type RefObject } from "react"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import type { AppSurface } from "@/contexts/TabContext"
import { addDocument, loadFileViewerText, uploadFile } from "@/lib/backend/api"
import { allDocumentsResource, projectDocumentsResource } from "@/lib/backend/resources"
import { attachToComposer } from "@/features/chat/attachToComposer"
import { architectureSource, isArchitecturePath } from "@/lib/domain/architecturePaths"
import { renderCanvasTextToJpeg } from "@/lib/domain/canvasRender"
import type { ProjectDocument } from "@/lib/domain/projects"
import { baseName } from "@/lib/paths"
import { errorMessage } from "@/lib/utils"
import { useEventListener } from "@/hooks/useEventListener"
import { useResource, useResourceFamily } from "@/hooks/useResource"

/** A list while it loads. */
const NO_DOCUMENTS: ProjectDocument[] = []

/** Own the document picker: project and global lists, selection into chat attachments, and the Alt+X shortcut. */
export function useProjectDocuments({
  isActive,
  appMode,
  projectSlug,
  chatId,
  chatInputRef,
  track,
}: {
  isActive: boolean
  appMode: AppSurface
  projectSlug: string | null
  chatId: string
  chatInputRef: RefObject<ChatInputHandle | null>
  track: <T>(work: () => Promise<T>) => Promise<T>
}) {
  const [documentOpen, setDocumentOpen] = useState(false)
  // The sidebar's shared lists, which change events keep current. A failed read leaves the picker empty without the startup banner.
  const projectDocuments = useResourceFamily(null, projectDocumentsResource, projectSlug ? [projectSlug] : [])[projectSlug ?? ""] ?? NO_DOCUMENTS
  const allDocuments = useResource(null, allDocumentsResource, documentOpen) ?? NO_DOCUMENTS
  const [architectureError, setArchitectureError] = useState("")

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
      await attachToComposer(chatInputRef, track, path)
    }
  }, [chatInputRef, track, chatId, projectSlug])

  const openDocuments = useCallback(() => setDocumentOpen(true), [])

  // One text attachment holding every `.c4` source as a fenced LikeC4 block under its workspace path.
  const handleAttachArchitecture = useCallback(async () => {
    if (!projectSlug) return
    setArchitectureError("")
    try {
      await track(async () => {
        const texts = await Promise.all(architectureSources.map(async (source) => ({ path: architectureSource(source.path), text: await loadFileViewerText(source.path) })))
        const content = `# ${projectSlug} architecture\n\n${texts.map(({ path, text }) => `## ${path}\n\n\`\`\`likec4\n${text}\n\`\`\``).join("\n\n")}`
        const file = new File([content], `${projectSlug}-architecture-${crypto.randomUUID().slice(0, 8)}.txt`, { type: "text/plain" })
        const attachment = await uploadFile(chatId, file, projectSlug)
        chatInputRef.current?.addAttachment({ ...attachment, content })
      })
    } catch (cause) {
      setArchitectureError(errorMessage(cause, "Could not attach architecture"))
    }
  }, [projectSlug, architectureSources, chatId, chatInputRef, track])

  // The project's catalog change updates its list, a failed add leaves it as it was.
  const handleAddDocToProject = useCallback(async (path: string) => {
    if (projectSlug) await addDocument(projectSlug, path).catch(() => {})
  }, [projectSlug])

  useEventListener("window", "keydown", (e) => {
    // Alt+X toggles the document picker
    if (e.altKey && (e.key === "x" || e.key === "X")) {
      e.preventDefault()
      setDocumentOpen((open) => !open)
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
    openDocuments,
    handleDocumentSelect,
    handleAddDocToProject,
  }
}
