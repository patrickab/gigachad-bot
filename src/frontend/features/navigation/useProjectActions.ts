"use client"

import { useState } from "react"
import { useProject } from "@/contexts/ProjectContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import type { ProjectListItem } from "@/lib/domain/projects"
import { useBranches } from "./BranchContext"

/** Rename, archive and delete of sidebar projects, with the rename dialog state. */
export function useProjectActions(onProjectRenamed?: (oldSlug: string, newSlug: string) => void) {
  const { deleteProject, setProjectArchived, renameProject } = useProject()
  const { refreshAll } = useBranches()
  const { schedule } = useUndoDelete()
  const { run: archive, pending: archivePending, error: archiveError } = useAsyncAction(setProjectArchived, "Could not update project archive")
  const [renameTarget, setRenameTarget] = useState<ProjectListItem | null>(null)
  const { run: rename, pending: renamePending, error: renameError, setError: setRenameError } = useAsyncAction(async (target: ProjectListItem, name: string) => {
    const slug = await renameProject(target.slug, name)
    if (slug !== target.slug) {
      onProjectRenamed?.(target.slug, slug)
      void refreshAll().catch(() => {})
    }
    setRenameTarget(null)
  }, "Could not rename project")

  const startRename = (project: ProjectListItem) => {
    setRenameError("")
    setRenameTarget(project)
  }

  /** Archive or restore `slug` unless another archive change is running. */
  const changeArchive = (slug: string, archived: boolean) => {
    if (!archivePending) void archive(slug, archived)
  }

  const removeProject = (project: ProjectListItem) => schedule(`project:${project.slug}`, project.name, () => deleteProject(project.slug))

  /** Props of the rename `SaveAsModal`. */
  const renameDialog = {
    open: renameTarget != null,
    onClose: () => { if (!renamePending) setRenameTarget(null) },
    initialName: renameTarget?.name,
    pending: renamePending,
    error: renameError,
    onSave: (name: string) => { if (renameTarget && !renamePending) void rename(renameTarget, name) },
  }

  return { archiveError, changeArchive, startRename, removeProject, renameDialog }
}
