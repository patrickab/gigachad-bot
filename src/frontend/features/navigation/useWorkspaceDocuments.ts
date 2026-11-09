"use client"

import { useCallback, useEffect, useEffectEvent, useMemo, useState } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { useTabActive } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import type { FileTreeCreateChoice } from "@/components/ui/file-tree/FileTree"
import { addRootFilesToProject, createArchitectureSource, createWorkbenchDocument, listNotes, listProjectDocuments, loadTemplateSets, moveDocument, moveDocumentToFolder, promoteToTexProject, removeDocument, writeBinaryDocument, writeDocument } from "@/lib/backend/api"
import { announceFileRemoval, onFileChange, onFileRemoval } from "@/lib/backend/fileEvents"
import { subscribeToChanges } from "@/lib/backend/syncStream"
import { isNewFileIn } from "@/lib/domain/files"
import type { TemplateSetInfo } from "@/lib/domain/latex"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"
import type { ProjectDocument, ProjectListItem } from "@/lib/domain/projects"
import { opensInWorkbench } from "@/lib/domain/workbench"
import { isInside } from "@/lib/paths"
import { errorMessage } from "@/lib/utils"

/** Key of a project's catalog entry, which lists its files. */
const PROJECT_FILE_LIST = /^chat\/[^/]+\/project\.json$/

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

/** Whether `path` is a file stored in a project (not a canvas or a file on disk), the kind that can move into a folder. */
export const isStoredProjectFile = (path: string) => /^project\/[^/]+\/document\//.test(path) && !path.endsWith(".canvas")

/** `documents` without the entries at or inside the deleted `path`, the same array when none is. */
function withoutRemoved(documents: ProjectDocument[], path: string): ProjectDocument[] {
  const kept = documents.filter((doc) => !isInside(doc.path, path))
  return kept.length === documents.length ? documents : kept
}

/** Kinds the files folder's `+` creates. */
const FILE_KINDS = ["Markdown", "LaTeX"]

/** The LaTeX template choice for a plain `.tex` file, listed before the server's template sets. */
const BLANK_TEMPLATE = "Blank"

/** `name` as a LaTeX file name: `.tex` appended unless it has it. */
const latexName = (name: string) => (name.toLowerCase().endsWith(".tex") ? name : `${name}.tex`)

/** `name` as a Markdown file name: `.md` appended unless it names its own extension. */
const markdownName = (name: string) => (/\.[^./]+$/.test(name) ? name : `${name}.md`)

interface WorkspaceDocumentsOptions {
  projects: ProjectListItem[]
  /** A change reads both lists again while the tab is active. */
  reloadKey: string
  moveHistory: (path: string, scope: string) => Promise<void>
  onDocumentsChanged?: () => void
  onDocumentSelect?: (path: string, scope: string) => void
  /** Opens `.tex .bib .sty .cls` and `.md` files in the workbench, absent where it is unavailable. */
  onWorkbenchSelect?: (path: string, scope: string) => void
  onCanvasSelect?: (path: string, scope: string) => void
  onCanvasDeleted?: (path: string) => void
  onArchitectureSelect?: (slug: string) => void
}

/** Loose notes and project documents of the sidebar, with the actions that change them. */
export function useWorkspaceDocuments({
  projects, reloadKey, moveHistory, onDocumentsChanged, onDocumentSelect, onWorkbenchSelect, onCanvasSelect, onCanvasDeleted, onArchitectureSelect,
}: WorkspaceDocumentsOptions) {
  const isActive = useTabActive()
  const { schedule } = useUndoDelete()
  const [documents, setDocuments] = useState<Record<string, ProjectDocument[]>>({})
  const [notes, setNotes] = useState<ProjectDocument[]>([])
  /** Project whose create-document panel is open. */
  const [createTarget, setCreateTarget] = useState<string | null>(null)
  const [fileError, setFileError] = useState("")
  const [templateSets, setTemplateSets] = useState<TemplateSetInfo[]>([])

  // The LaTeX templates come from the template sets on the server, none when they fail to load.
  // Keyed on availability, not on the callback: a parent's new callback identity would refetch on every render.
  const workbenchAvailable = onWorkbenchSelect !== undefined
  useEffect(() => {
    if (!workbenchAvailable) return
    let cancelled = false
    loadTemplateSets().then((sets) => { if (!cancelled) setTemplateSets(sets) }, () => {})
    return () => { cancelled = true }
  }, [workbenchAvailable])
  const templateNames = useMemo(() => new Map(templateSets.map((set) => [set.label, set.name])), [templateSets])
  /** The `+` choice of the files folder: the file type, then the template for LaTeX. Absent without the workbench, files are Markdown then. */
  const newFileChoice = useMemo((): FileTreeCreateChoice | undefined => {
    if (!onWorkbenchSelect) return undefined
    const templates = { label: "Template", options: [BLANK_TEMPLATE, ...templateNames.keys()], initial: BLANK_TEMPLATE, floating: true }
    return { label: "File type", options: FILE_KINDS, initial: "Markdown", floating: true, then: templateNames.size ? { LaTeX: templates } : undefined }
  }, [onWorkbenchSelect, templateNames])

  const loadProjectDocuments = useCallback(async (signal?: AbortSignal) => {
    const entries = await Promise.all(projects.map(async (project) =>
      [project.slug, await listProjectDocuments(project.slug, signal)] as const))
    return Object.fromEntries(entries)
  }, [projects])

  useStartupRead("Sidebar notes", listNotes, setNotes, isActive, reloadKey)
  useStartupRead("Sidebar project documents", loadProjectDocuments, setDocuments, isActive, reloadKey)

  // A project's file list changes without this device when an extraction finishes, or on another device.
  useEffect(() => subscribeToChanges((event) => {
    if (!isActive || event.resource_kind !== "document" || !PROJECT_FILE_LIST.test(event.resource_key)) return
    void loadProjectDocuments().then(setDocuments).catch(() => {})
  }), [isActive, loadProjectDocuments])

  // A file created inside a folder entry (a pasted image, a file made on another device) shows up in its tree.
  const folderEntries = useMemo(() => Object.values(documents).flat().filter((doc) => doc.children), [documents])
  const isNewInFolders = useEffectEvent((path: string) => isActive && isNewFileIn(folderEntries, path))
  const reloadDocuments = useEffectEvent(() => { void loadProjectDocuments().then(setDocuments).catch(() => {}) })
  useEffect(() => onFileChange((path) => isNewInFolders(path), () => reloadDocuments()), [])

  // A file or folder deleted here or on another device leaves the lists of every tab's sidebar at once, the lists reload when it lay in a folder entry.
  const dropRemoved = useEffectEvent((path: string) => {
    setNotes((current) => withoutRemoved(current, path))
    setDocuments((current) => {
      const next = Object.fromEntries(Object.entries(current).map(([slug, docs]) => [slug, withoutRemoved(docs, path)]))
      return Object.keys(next).every((slug) => next[slug] === current[slug]) ? current : next
    })
    if (isActive && folderEntries.some((entry) => isInside(path, entry.path))) reloadDocuments()
  })
  useEffect(() => onFileRemoval((path) => dropRemoved(path)), [])

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

  /** Open project file `path`: a workbench file (`.tex .bib .sty .cls`, `.md`) in the workbench where available, notes and the rest in the document editor. */
  const selectDocument = (path: string, scope: string) => {
    if (onWorkbenchSelect && opensInWorkbench(path)) onWorkbenchSelect(path, scope)
    else onDocumentSelect?.(path, scope)
  }

  /**
   * Create a `kind` file `name` in project `scope` and open it, a LaTeX file from the template set labelled `template`.
   *
   * - LaTeX lives on disk: a plain `.tex` for `BLANK_TEMPLATE` or no template, a folder with `main.tex` for a template set
   * - Markdown and unknown kinds live in the project until the workbench opens them
   */
  const createFile = async (scope: string, name: string, kind: string, template?: string) => {
    if (onWorkbenchSelect && kind === "LaTeX") {
      const path = await createWorkbenchDocument(scope, latexName(name), template && templateNames.get(template))
      await refresh()
      onWorkbenchSelect(path, scope)
    } else {
      await createAndOpen(scope, markdownName(name), (path) => selectDocument(path, scope))
    }
  }

  /** Move a canvas, stored project file or chat history `path` into `scope`; architecture sources stay. */
  const move = async (path: string, scope: string) => {
    if (isArchitecturePath(path)) return
    if (path.endsWith(".canvas") || isStoredProjectFile(path)) {
      const from = Object.entries(documents).find(([, docs]) => docs.some((doc) => doc.path === path))?.[0] ?? ""
      if (from !== scope) {
        await moveDocument(path, from, scope)
        await refresh()
        onDocumentsChanged?.()
        onCanvasDeleted?.(path)
      }
    } else {
      await moveHistory(path, scope)
    }
  }

  /** Move the stored project file `path` into the mounted `folder`, where it then behaves as a mounted file. */
  const moveToFolder = async (path: string, folder: string) => {
    await moveDocumentToFolder(path, folder)
    await refresh()
    onDocumentsChanged?.()
    onCanvasDeleted?.(path)
  }

  /** Turn project `scope`'s workbench `.tex` at `path` into its LaTeX project folder, which the list then shows as a folder. */
  const promote = async (path: string, scope: string) => {
    setFileError("")
    try {
      await promoteToTexProject(scope, path)
    } catch (cause) {
      setFileError(errorMessage(cause, "Could not make a LaTeX project"))
    }
    await refresh()
    onDocumentsChanged?.()
  }

  /** Pick files from this device and store them as documents of project `scope`. */
  const uploadFiles = async (scope: string) => {
    const files = await pickFiles()
    if (files.length === 0) return
    setFileError("")
    try {
      await Promise.all(files.map((file) => writeBinaryDocument(scope, file.name, file)))
    } catch (cause) {
      setFileError(errorMessage(cause, "Could not upload files"))
    }
    await refresh()
    onDocumentsChanged?.()
  }

  /** Delete `document` after the undo window, with the `alsoRemove` documents that hang under it (a PDF's extraction). */
  const remove = ({ path, name }: ProjectDocument, scope: string, alsoRemove: ProjectDocument[] = []) => {
    schedule(`document:${scope}:${path}`, name, async () => {
      for (const target of [path, ...alsoRemove.map((extra) => extra.path)]) {
        for (const removed of await removeDocument(scope, target)) announceFileRemoval(removed)
      }
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
        open = path.endsWith(".canvas") ? () => onCanvasSelect?.(path, slug) : () => selectDocument(path, slug)
      }
      setCreateTarget(null)
      await refresh()
      onDocumentsChanged?.()
      open()
    } catch { /* the panel stays open to retry */ }
  }

  return {
    documents, notes, refresh, createAndOpen, selectDocument, createFile, newFileChoice, move, moveToFolder, promote, remove, importFiles,
    uploadFiles, fileError, hasArchitecture, openArchitecture, createTarget, setCreateTarget, createDocument,
  }
}
