"use client"

import { useMemo, useState } from "react"
import { useTabActive } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import type { FileTreeCreateChoice } from "@/components/ui/file-tree/FileTree"
import { useResource, useResourceFamily } from "@/hooks/useResource"
import { addRootFilesToProject, createArchitectureSource, createWorkbenchDocument, moveDocument, moveDocumentToFolder, promoteToTexProject, removeDocument, writeBinaryDocument, writeDocument } from "@/lib/backend/api"
import { announceFileRemoval } from "@/lib/backend/fileEvents"
import { notesResource, projectDocumentsResource, refreshDocumentLists, refreshProjectDocuments, templateSetsResource } from "@/lib/backend/resources"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"
import type { ProjectDocument, ProjectListItem } from "@/lib/domain/projects"
import { opensInWorkbench } from "@/lib/domain/workbench"
import { errorMessage } from "@/lib/utils"

/** The notes while they load. */
const NO_NOTES: ProjectDocument[] = []

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
  moveHistory: (path: string, scope: string) => Promise<void>
  onDocumentSelect?: (path: string, scope: string) => void
  /** Opens `.tex .bib .sty .cls` and `.md` files in the workbench, absent where it is unavailable. */
  onWorkbenchSelect?: (path: string, scope: string) => void
  onCanvasSelect?: (path: string, scope: string) => void
  onCanvasDeleted?: (path: string) => void
  onArchitectureSelect?: (slug: string) => void
}

/** Loose notes and project documents of the sidebar, with the actions that change them. Every tab shares the lists, `watchDocumentLists` keeps them current. */
export function useWorkspaceDocuments({
  projects, moveHistory, onDocumentSelect, onWorkbenchSelect, onCanvasSelect, onCanvasDeleted, onArchitectureSelect,
}: WorkspaceDocumentsOptions) {
  const { schedule } = useUndoDelete()
  const isActive = useTabActive()
  const documents = useResourceFamily("Sidebar project documents", projectDocumentsResource, projects.map((project) => project.slug))
  const notes = useResource("Sidebar notes", notesResource) ?? NO_NOTES
  /** Project whose create-document panel is open. */
  const [createTarget, setCreateTarget] = useState<string | null>(null)
  const [fileError, setFileError] = useState("")

  // The LaTeX templates come from the template sets on the server, none when they fail to load. The tab's next activation tries again.
  const templateSets = useResource(null, templateSetsResource, onWorkbenchSelect !== undefined && isActive)
  const templateNames = useMemo(() => new Map((templateSets ?? []).map((set) => [set.label, set.name])), [templateSets])
  /** The `+` choice of the files folder: the file type, then the template for LaTeX. Absent without the workbench, files are Markdown then. */
  const newFileChoice = useMemo((): FileTreeCreateChoice | undefined => {
    if (!onWorkbenchSelect) return undefined
    const templates = { label: "Template", options: [BLANK_TEMPLATE, ...templateNames.keys()], initial: BLANK_TEMPLATE, floating: true }
    return { label: "File type", options: FILE_KINDS, initial: "Markdown", floating: true, then: templateNames.size ? { LaTeX: templates } : undefined }
  }, [onWorkbenchSelect, templateNames])

  /** Create `name` in `scope`, read its list again and hand the new path to `open`. */
  const createAndOpen = async (scope: string, name: string, open: (path: string) => void) => {
    const { path } = await writeDocument(scope, name)
    await refreshProjectDocuments(scope)
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
      await refreshProjectDocuments(scope)
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
        await refreshDocumentLists()
        onCanvasDeleted?.(path)
      }
    } else {
      await moveHistory(path, scope)
    }
  }

  /** Move the stored project file `path` into the mounted `folder`, where it then behaves as a mounted file. */
  const moveToFolder = async (path: string, folder: string) => {
    await moveDocumentToFolder(path, folder)
    await refreshDocumentLists()
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
    await refreshDocumentLists()
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
    await refreshProjectDocuments(scope)
  }

  /** Delete `document` after the undo window, with the `alsoRemove` documents that hang under it (a PDF's extraction). */
  const remove = ({ path, name }: ProjectDocument, scope: string, alsoRemove: ProjectDocument[] = []) => {
    schedule(`document:${scope}:${path}`, name, async () => {
      for (const target of [path, ...alsoRemove.map((extra) => extra.path)]) {
        for (const removed of await removeDocument(scope, target)) announceFileRemoval(removed)
      }
      await refreshDocumentLists()
      onCanvasDeleted?.(path)
    })
  }

  const importFiles = async (scope: string, paths: string[]) => {
    await addRootFilesToProject(scope, paths)
    await refreshProjectDocuments(scope)
  }

  const hasArchitecture = (slug: string) => (documents[slug] ?? []).some((doc) => isArchitecturePath(doc.path))

  const openArchitecture = async (slug: string) => {
    if (!hasArchitecture(slug)) {
      await createArchitectureSource(slug, "system.c4")
      await refreshProjectDocuments(slug)
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
      await refreshProjectDocuments(slug)
      open()
    } catch { /* the panel stays open to retry */ }
  }

  return {
    documents, notes, createAndOpen, selectDocument, createFile, newFileChoice, move, moveToFolder, promote, remove, importFiles,
    uploadFiles, fileError, hasArchitecture, openArchitecture, createTarget, setCreateTarget, createDocument,
  }
}
