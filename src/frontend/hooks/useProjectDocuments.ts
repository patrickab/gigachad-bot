"use client"

import { useCallback, useEffect, useMemo, useState, type Dispatch, type RefObject, type SetStateAction } from "react"
import type { ChatInputHandle } from "@/components/ChatInput"
import type { AppSurface } from "@/contexts/SidebarContext"
import {
  addDocument,
  attachDocument,
  attachFileVaultFile,
  listAllDocuments,
  listProjectDocuments,
  listProjectVaultDocuments,
  loadFileViewerText,
  removeDocument,
  uploadDocument,
  uploadFile,
  writeDocument,
} from "@/lib/api"
import { buildHiddenContent } from "@/lib/attachments"
import { renderCanvasToJpeg } from "@/lib/drawing"
import type { Message, ProjectDocument } from "@/lib/types"

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
  const [documentOpen, setDocumentOpen] = useState(false)
  const [createDocOpen, setCreateDocOpen] = useState(false)
  const [projectDocuments, setProjectDocuments] = useState<ProjectDocument[]>([])
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
  // into the cloud library) appears exactly once — library/cloud wins.
  const mergedDocuments = useMemo(() => {
    const seen = new Set<string>()
    const out: ProjectDocument[] = []
    for (const d of projectDocuments) {
      if (seen.has(d.path)) continue
      seen.add(d.path)
      out.push(d)
    }
    for (const d of vaultProjectDocs) {
      if (seen.has(d.path)) continue
      seen.add(d.path)
      out.push(d)
    }
    return out
  }, [projectDocuments, vaultProjectDocs])

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
        const doc = text.trim() ? JSON.parse(text) : null
        if (!doc?.strokes?.length && !doc?.texts?.length) return
        const blob = await renderCanvasToJpeg(doc.strokes ?? [], 20, [], doc.texts ?? [])
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
      await writeDocument(activeProject, name)
      setCreateDocOpen(false)
      refreshDocuments()
    } catch { /* */ }
  }, [activeProject, refreshDocuments])

  const handleDeleteDocument = useCallback(async (path: string) => {
    if (!activeProject) return
    try {
      await removeDocument(activeProject, path)
      refreshDocuments()
    } catch { /* */ }
  }, [activeProject, refreshDocuments])

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
    refreshDocuments,
    openDocuments,
    handleDocumentSelect,
    handleCreateDocument,
    handleDeleteDocument,
    handleDocumentSaved,
    handleDocumentUpload,
    handleAddDocToProject,
  }
}
