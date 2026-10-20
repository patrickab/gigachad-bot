"use client"

import { useCallback, useMemo, useState } from "react"
import { Plus, Layers, FolderOpen } from "lucide-react"
import type { ProjectDocument } from "@/lib/domain/types"
import { FileViewer } from "@/features/editor/FileViewer"

type Scope = "project" | "global"

interface DocumentPickerProps {
  projectSlug: string | null
  projectDocuments: ProjectDocument[]
  allDocuments: ProjectDocument[]
  onAttach: (path: string) => void
  onAddToProject?: (path: string) => void
  onClose: () => void
}

/** Searchable document list scoped to the project or all documents. Left arrow toggles scope. */
export function DocumentPicker({
  projectSlug,
  projectDocuments,
  allDocuments,
  onAttach,
  onAddToProject,
  onClose,
}: DocumentPickerProps) {
  const [scope, setScope] = useState<Scope>(projectSlug ? "project" : "global")

  const docs = scope === "project" ? projectDocuments : allDocuments
  const files = useMemo(() => docs.map((d) => d.path), [docs])
  const projectPaths = useMemo(() => new Set(projectDocuments.map((d) => d.path)), [projectDocuments])
  const statusLabel = scope === "project" ? (
    <><FolderOpen className="h-3 w-3 shrink-0" /><span className="truncate">project — {projectSlug}</span></>
  ) : (
    <><Layers className="h-3 w-3 shrink-0" /><span className="truncate">all documents</span></>
  )

  // Hover "+" on global rows to copy a document into the current project
  const renderRowSuffix = useCallback((path: string) => {
    if (scope !== "global" || !onAddToProject || !projectSlug || projectPaths.has(path)) return null
    return (
      <span
        role="button"
        tabIndex={-1}
        onClick={(e) => { e.stopPropagation(); onAddToProject(path) }}
        className="rounded p-0.5 text-ink-faint opacity-0 group-hover:opacity-100 hover:text-ink hover:bg-surface transition"
      >
        <Plus className="h-3 w-3" />
      </span>
    )
  }, [scope, onAddToProject, projectSlug, projectPaths])

  return (
    <FileViewer
      files={files}
      onSelect={onAttach}
      onClose={onClose}
      placeholder="Search documents…"
      emptyLabel="No documents"
      statusLabel={statusLabel}
      renderRowSuffix={renderRowSuffix}
      onArrowLeft={projectSlug ? () => setScope((s) => (s === "project" ? "global" : "project")) : undefined}
    />
  )
}
