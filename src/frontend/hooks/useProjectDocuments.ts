"use client"

import { useCallback, useEffect, useMemo, useState, type Dispatch, type RefObject, type SetStateAction } from "react"
import type { ChatInputHandle } from "@/components/ChatInput"
import type { AppSurface } from "@/contexts/SidebarContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { parseCanvasDoc, type CanvasAttachment, type CanvasDocument, type CanvasFrame } from "@/components/CanvasEditor"
import {
  addDocument,
  createArchitectureSource,
  deleteArchitectureSources,
  attachDocument,
  attachFileVaultFile,
  fileViewerRawUrl,
  listAllDocuments,
  listProjectDocuments,
  listProjectVaultDocuments,
  loadFileViewerText,
  removeDocument,
  uploadDocument,
  uploadFile,
  renameDocument,
  writeDocument,
} from "@/lib/api"
import { architectureSource, isArchitecturePath } from "@/lib/architecture"
import { buildHiddenContent } from "@/lib/attachments"
import { renderCanvasToJpeg, type EmbedRect } from "@/lib/drawing"
import type { Message, ProjectDocument } from "@/lib/types"
 
const IMAGE_RENDER_EXCLUDED_ATTACHMENT_KINDS: Partial<Record<CanvasAttachment["kind"], true>> = { pdf: true }
const FALLBACK_ASPECT = 1.3

function imageEmbed(path: string, { x, y, width, height }: Pick<CanvasFrame | CanvasAttachment, "x" | "y" | "width"> & { height?: number }): EmbedRect {
  return { url: fileViewerRawUrl(path), x, y, width, aspect: height ? height / width : FALLBACK_ASPECT }
}
function renderableCanvasImages({ frames = [], attachments = [] }: Partial<Pick<CanvasDocument, "frames" | "attachments">>): EmbedRect[] {
  return [
    ...frames.filter((frame) => frame.kind === "image" && frame.path).map((frame) => imageEmbed(frame.path!, frame)),
    ...attachments
      .filter((attachment) => attachment.path && !IMAGE_RENDER_EXCLUDED_ATTACHMENT_KINDS[attachment.kind])
      .map((attachment) => imageEmbed(attachment.path!, attachment)),
  ]
}

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
  const [vaultProjectDocs, setVaultProjectDocs] = useState<ProjectDocument[]>([])
  const [allDocuments, setAllDocuments] = useState<ProjectDocument[]>([])

  // A path is "vault-surfaced" only when it's not already a project file — a
  // promoted vault PDF that's also in the project behaves as a project doc
  // (deletable, library attach), and the merged list shows it just once.
  const vaultDocPaths = useMemo(() => {
    const projectPaths = new Set(projectDocuments.map((d) => d.path))
    return new Set(vaultProjectDocs.filter((d) => !projectPaths.has(d.path)).map((d) => d.path))
  }, [projectDocuments, vaultProjectDocs])
  // Merge library/project docs with vault docs, deduping by path so a PDF that
  // exists both in the project files and in a mounted vault (or was promoted
  // into the cloud library) appears exactly once — library/cloud wins. Documents
  // whose deletion is pending (`handleDeleteDocument`) are hidden.
  const mergedDocuments = useMemo(() => {
    const seen = new Set<string>()
    const out: ProjectDocument[] = []
    for (const d of [...projectDocuments, ...vaultProjectDocs]) {
      if (seen.has(d.path) || pending(`document:${activeProject}:${d.path}`)) continue
      seen.add(d.path)
      out.push(d)
    }
    return out
  }, [projectDocuments, vaultProjectDocs, activeProject, pending])
  const architectureSources = useMemo(() => projectDocuments.filter((document) => isArchitecturePath(document.path)), [projectDocuments])

  const handleDocumentSelect = useCallback(async (path: string) => {
    setDocumentOpen(false)
    if (vaultDocPaths.has(path)) {
      setExtracting((n) => n + 1)
      try { chatInputRef.current?.addAttachment(await attachFileVaultFile(path)) } catch { /* */ }
      finally { setExtracting((n) => n - 1) }
      return
    }
    if (path.endsWith(".canvas")) {
      try {
        const live = liveCanvasRef.current
        const text = live?.path === path ? live.content : await loadFileViewerText(path)
        const doc = text.trim() ? parseCanvasDoc(text) : null
        const strokes = doc?.strokes ?? []
        const texts = doc?.texts ?? []
        const images = doc ? renderableCanvasImages(doc) : []
        if (!doc || (!strokes.length && !texts.length && images.length === 0)) return
        const blob = await renderCanvasToJpeg(strokes, 20, images, texts)
        const name = path.split("/").pop()!.replace(/\.canvas$/, ".jpg")
        const file = new File([blob], name, { type: "image/jpeg" })
        const att = await uploadFile(chatId, file, activeProject, true)
        chatInputRef.current?.addAttachment(att)
      } catch { /* */ }
    } else {
      setExtracting((n) => n + 1)
      try { chatInputRef.current?.addAttachment(await attachDocument(chatId, path, activeProject)) } catch { /* */ }
      finally { setExtracting((n) => n - 1) }
    }
  }, [chatId, activeProject, vaultDocPaths, chatInputRef, liveCanvasRef, setExtracting])

  const loadProjectDocs = useCallback(() => {
    if (activeProject) {
      listProjectDocuments(activeProject).then(setProjectDocuments).catch(() => {})
      listProjectVaultDocuments(activeProject).then(setVaultProjectDocs).catch(() => setVaultProjectDocs([]))
    } else {
      setProjectDocuments([])
      setVaultProjectDocs([])
    }
  }, [activeProject])

  const refreshDocuments = useCallback(() => {
    loadProjectDocs()
    listAllDocuments().then(setAllDocuments).catch(() => {})
  }, [loadProjectDocs])

  useEffect(() => { loadProjectDocs() }, [loadProjectDocs])

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
    schedule(`document:${slug}:${path}`, path.split("/").pop() ?? path, async () => {
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
        const attachments = msg.attachments!.map(a => a.name === filename ? { ...a, content } : a)
        return { ...msg, attachments, hiddenContent: buildHiddenContent(attachments) || undefined }
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
    if (!activeProject || vaultDocPaths.has(path) || isArchitecturePath(path)) return
    try {
      await renameDocument(activeProject, path, name)
      refreshDocuments()
    } catch { /* the stored name stays authoritative when rename fails */ }
  }, [activeProject, vaultDocPaths, refreshDocuments])

  useEffect(() => {
    if (!isActive || appMode === "canvas") return
    const onKey = (e: KeyboardEvent) => {
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
    vaultDocPaths,
    mergedDocuments,
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
