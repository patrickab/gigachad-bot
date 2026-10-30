"use client"

import { useCallback, useState } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { useTabActive } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { addRootFilesToProject, createArchitectureSource, listNotes, listProjectDocuments, moveDocument, removeDocument, writeBinaryDocument, writeDocument } from "@/lib/backend/api"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"
import type { ProjectDocument, ProjectListItem } from "@/lib/domain/projects"
import { errorMessage } from "@/lib/utils"

/** Open the OS file dialog (any platform, the webview provides it) and resolve with the chosen files, none when cancelled. */
function pickFiles(): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement("input")
    input.type = "file"
    input.multiple = true
    input.onchange = () => resolve(Array.from(input.files ?? []))
    input.oncancel = () => resolve([])
    input.click()
  })
}

interface WorkspaceDocumentsOptions {
  projects: ProjectListItem[]
  /** A change reads both lists again while the tab is active. */
  reloadKey: string
  moveHistory: (path: string, scope: string) => Promise<void>
  onDocumentsChanged?: () => void
  onDocumentSelect?: (path: string, scope: string) => void
  onCanvasSelect?: (path: string, scope: string) => void
  onCanvasDeleted?: (path: string) => void
  onArchitectureSelect?: (slug: string) => void
}

/** Loose notes and project documents of the sidebar, with the actions that change them. */
export function useWorkspaceDocuments({
  projects, reloadKey, moveHistory, onDocumentsChanged, onDocumentSelect, onCanvasSelect, onCanvasDeleted, onArchitectureSelect,
}: WorkspaceDocumentsOptions) {
  const isActive = useTabActive()
  const { schedule } = useUndoDelete()
  const [documents, setDocuments] = useState<Record<string, ProjectDocument[]>>({})
  const [notes, setNotes] = useState<ProjectDocument[]>([])
  /** Project whose create-document panel is open. */
  const [createTarget, setCreateTarget] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState("")

  const loadProjectDocuments = useCallback(async (signal?: AbortSignal) => {
    const entries = await Promise.all(projects.map(async (project) =>
      [project.slug, await listProjectDocuments(project.slug, signal)] as const))
    return Object.fromEntries(entries)
  }, [projects])

  useStartupRead("Sidebar notes", listNotes, setNotes, isActive, reloadKey)
  useStartupRead("Sidebar project documents", loadProjectDocuments, setDocuments, isActive, reloadKey)

  const refresh = async () => {
    const [loose, entries] = await Promise.all([listNotes(), loadProjectDocuments()])
    setNotes(loose)
    setDocuments(entries)
  }

  /** Create `name` in `scope`, refresh the lists and hand the new path to `open`. */
  const createAndOpen = async (scope: string, name: string, open: (path: string) => void) => {
    const { path } = await writeDocument(scope, name)
    await refresh()
    open(path)
  }

  /** Move a canvas or chat history `path` into `scope`; architecture sources stay. */
  const move = async (path: string, scope: string) => {
    if (isArchitecturePath(path)) return
    if (path.endsWith(".canvas")) {
      const from = Object.entries(documents).find(([, docs]) => docs.some((doc) => doc.path === path))?.[0] ?? ""
      if (from !== scope) {
        await moveDocument(path, from, scope)
        await refresh()
        onCanvasDeleted?.(path)
      }
    } else {
      await moveHistory(path, scope)
    }
  }

  /** Pick files from this device and store them as documents of project `scope`. */
  const uploadFiles = async (scope: string) => {
    const files = await pickFiles()
    if (files.length === 0) return
    setUploadError("")
    try {
      await Promise.all(files.map((file) => writeBinaryDocument(scope, file.name, file)))
    } catch (cause) {
      setUploadError(errorMessage(cause, "Could not upload files"))
    }
    await refresh()
    onDocumentsChanged?.()
  }

  const remove = ({ path, name }: ProjectDocument, scope: string) => {
    schedule(`document:${scope}:${path}`, name, async () => {
      await removeDocument(scope, path)
      await refresh()
      onDocumentsChanged?.()
      onCanvasDeleted?.(path)
    })
  }

  const importFiles = async (scope: string, paths: string[]) => {
    await addRootFilesToProject(scope, paths)
    await refresh()
    onDocumentsChanged?.()
  }

  const hasArchitecture = (slug: string) => (documents[slug] ?? []).some((doc) => isArchitecturePath(doc.path))

  const openArchitecture = async (slug: string) => {
    if (!hasArchitecture(slug)) {
      await createArchitectureSource(slug, "system.c4")
      await refresh()
    }
    onArchitectureSelect?.(slug)
  }

  /** Create `name` in the create target and open it: architecture sources in the architecture window, canvases as canvas, the rest in the editor. */
  const createDocument = async (name: string) => {
    const slug = createTarget
    if (!slug) return
    try {
      let open: () => void
      if (name.endsWith(".c4")) {
        await createArchitectureSource(slug, name)
        open = () => onArchitectureSelect?.(slug)
      } else {
        const { path } = await writeDocument(slug, name)
        open = path.endsWith(".canvas") ? () => onCanvasSelect?.(path, slug) : () => onDocumentSelect?.(path, slug)
      }
      setCreateTarget(null)
      await refresh()
      onDocumentsChanged?.()
      open()
    } catch { /* the panel stays open to retry */ }
  }

  return { documents, notes, createAndOpen, move, remove, importFiles, uploadFiles, uploadError, hasArchitecture, openArchitecture, createTarget, setCreateTarget, createDocument }
}
