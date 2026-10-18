"use client"

import { useCallback, useEffect, useMemo, useState, type Dispatch, type RefObject, type SetStateAction } from "react"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import type { AppSurface } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import {
  addDocument,
  createArchitectureSource,
  deleteArchitectureSources,
  attachDocument,
  listAllDocuments,
  listProjectDocuments,
  loadFileViewerText,
  removeDocument,
  uploadDocument,
  uploadFile,
  renameDocument,
  writeDocument,
} from "@/lib/backend/api"
import { architectureSource, isArchitecturePath } from "@/lib/domain/architecturePaths"
import { withAttachments } from "@/features/chat/attachments"
import { renderCanvasTextToJpeg } from "@/lib/domain/canvasRender"
import type { Message, ProjectDocument } from "@/lib/domain/types"
import { baseName } from "@/lib/paths"

/** Own the project documents panel: lists, selection into chat attachments, CRUD with undoable delete, and the Alt+X shortcut. */
export function useProjectDocuments({
  isActive,
  appMode,
  activeProject,
  chatId,
  chatInputRef,
  liveCanvasRef,
  setExtracting,
  setMessages,
}: {
  isActive: boolean
  appMode: AppSurface
  activeProject: string | null
  chatId: string
  chatInputRef: RefObject<ChatInputHandle | null>
  liveCanvasRef: RefObject<{ path: string; content: string } | null>
  setExtracting: Dispatch<SetStateAction<number>>
  setMessages: Dispatch<SetStateAction<Message[]>>
}) {
  const { schedule, pending } = useUndoDelete()
  const [documentOpen, setDocumentOpen] = useState(false)
  const [createDocOpen, setCreateDocOpen] = useState(false)
  const [projectDocuments, setProjectDocuments] = useState<ProjectDocument[]>([])
  const [architectureError, setArchitectureError] = useState("")
  const [allDocuments, setAllDocuments] = useState<ProjectDocument[]>([])

  // Documents with a pending deletion (`handleDeleteDocument`) are hidden.
  const visibleDocuments = useMemo(
    () => projectDocuments.filter((d) => !pending(`document:${activeProject}:${d.path}`)),
    [projectDocuments, activeProject, pending],
  )
  const architectureSources = useMemo(() => projectDocuments.filter((document) => isArchitecturePath(document.path)), [projectDocuments])

  const handleDocumentSelect = useCallback(async (path: string) => {
    setDocumentOpen(false)
    // Canvas docs are attached as a rendered JPEG, preferring the unsaved live content
    if (path.endsWith(".canvas")) {
      try {
        const live = liveCanvasRef.current
        const text = live?.path === path ? live.content : await loadFileViewerText(path)
        const blob = await renderCanvasTextToJpeg(text)
        if (!blob) return
        const name = baseName(path).replace(/\.canvas$/, ".jpg")
        const file = new File([blob], name, { type: "image/jpeg" })
        const att = await uploadFile(chatId, file, activeProject, true)
        chatInputRef.current?.addAttachment(att)
      } catch { /* */ }
    } else {
      setExtracting((n) => n + 1)
      try { chatInputRef.current?.addAttachment(await attachDocument(chatId, path, activeProject)) } catch { /* */ }
      finally { setExtracting((n) => n - 1) }
    }
  }, [chatId, activeProject, chatInputRef, liveCanvasRef, setExtracting])

  const loadProjectDocs = useCallback(() => {
    if (activeProject) listProjectDocuments(activeProject).then(setProjectDocuments).catch(() => {})
  }, [activeProject])

  const refreshDocuments = useCallback(() => {
    loadProjectDocs()
    listAllDocuments().then(setAllDocuments).catch(() => {})
  }, [loadProjectDocs])

  useEffect(() => { loadProjectDocs() }, [loadProjectDocs])

  // Without a project there are no documents, whatever was loaded for the previous one
  if (!activeProject && projectDocuments.length > 0) setProjectDocuments([])

  const handleCreateDocument = useCallback(async (name: string) => {
    if (!activeProject) return
    try {
      // A `.c4` name may carry folders (`backend/api.c4`): it joins the project's architecture workspace, empty.
      if (name.endsWith(".c4")) await createArchitectureSource(activeProject, name)
      else await writeDocument(activeProject, name)
      setCreateDocOpen(false)
      refreshDocuments()
    } catch { /* */ }
  }, [activeProject, refreshDocuments])

  // One text attachment holding every `.c4` source as a fenced LikeC4 block under its workspace path.
  const handleAttachArchitecture = useCallback(async () => {
    if (!activeProject) return
    setArchitectureError("")
    setExtracting((count) => count + 1)
    try {
      const texts = await Promise.all(architectureSources.map(async (source) => ({ path: architectureSource(source.path), text: await loadFileViewerText(source.path) })))
      const content = `# ${activeProject} architecture\n\n${texts.map(({ path, text }) => `## ${path}\n\n\`\`\`likec4\n${text}\n\`\`\``).join("\n\n")}`
      const file = new File([content], `${activeProject}-architecture-${crypto.randomUUID().slice(0, 8)}.txt`, { type: "text/plain" })
      const attachment = await uploadFile(chatId, file, activeProject)
      chatInputRef.current?.addAttachment({ ...attachment, content })
    } catch (cause) {
      setArchitectureError(cause instanceof Error ? cause.message : "Could not attach architecture")
    } finally {
      setExtracting((count) => count - 1)
    }
  }, [activeProject, architectureSources, chatId, chatInputRef, setExtracting])

  // Hidden until its undo window closes; the commit then deletes and refreshes the lists.
  const handleDeleteDocument = useCallback((path: string) => {
    if (!activeProject) return
    const slug = activeProject
    schedule(`document:${slug}:${path}`, baseName(path), async () => {
      // Architecture sources are refused while another source still references them.
      if (isArchitecturePath(path)) await deleteArchitectureSources(slug, [architectureSource(path)])
      else await removeDocument(slug, path)
      refreshDocuments()
    })
  }, [activeProject, refreshDocuments, schedule])

  const handleDocumentSaved = useCallback((filename?: string, content?: string) => {
    refreshDocuments()
    if (filename && content !== undefined) {
      setMessages(prev => prev.map(msg => {
        if (!msg.attachments?.some(a => a.name === filename)) return msg
        const attachments = msg.attachments.map(a => a.name === filename ? { ...a, content } : a)
        return withAttachments(msg, attachments)
      }))
    }
  }, [refreshDocuments, setMessages])

  const openDocuments = useCallback(() => {
    refreshDocuments()
    setDocumentOpen(true)
  }, [refreshDocuments])

  const handleDocumentUpload = useCallback(async (files: File[]) => {
    if (!activeProject) return
    await Promise.allSettled(files.map(f => uploadDocument(activeProject, f)))
    refreshDocuments()
  }, [activeProject, refreshDocuments])

  const handleAddDocToProject = useCallback(async (path: string) => {
    if (!activeProject) return
    try {
      await addDocument(activeProject, path)
      refreshDocuments()
    } catch { }
  }, [activeProject, refreshDocuments])

  const handleRenameDocument = useCallback(async (path: string, name: string) => {
    // Architecture sources are named by their path inside the workspace; there is no rename for them.
    if (!activeProject || isArchitecturePath(path)) return
    try {
      await renameDocument(activeProject, path, name)
      refreshDocuments()
    } catch { /* the stored name stays authoritative when rename fails */ }
  }, [activeProject, refreshDocuments])

  useEffect(() => {
    if (!isActive || appMode !== "chat") return
    const onKey = (e: KeyboardEvent) => {
      // Alt+X toggles the documents panel
      if (e.altKey && (e.key === "x" || e.key === "X")) {
        e.preventDefault()
        setDocumentOpen((open) => {
          if (!open) refreshDocuments()
          return !open
        })
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [isActive, appMode, refreshDocuments])

  return {
    documentOpen,
    setDocumentOpen,
    createDocOpen,
    setCreateDocOpen,
    projectDocuments,
    allDocuments,
    visibleDocuments,
    architectureError,
    /** Absent while the project has no architecture sources to attach. */
    handleAttachArchitecture: architectureSources.length > 0 ? handleAttachArchitecture : undefined,
    refreshDocuments,
    openDocuments,
    handleDocumentSelect,
    handleCreateDocument,
    handleDeleteDocument,
    handleDocumentSaved,
    handleDocumentUpload,
    handleRenameDocument,
    handleAddDocToProject,
  }
}
