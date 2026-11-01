"use client"

import { useProject } from "@/contexts/ProjectContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import type { ProjectListItem } from "@/lib/domain/projects"
import { useBranches } from "./BranchContext"

/** Rename, archive and delete of sidebar projects. `renameProjectTo` rejects with the reason, e.g. a taken name. */
export function useProjectActions(onProjectRenamed?: (oldSlug: string, newSlug: string) => void) {
  const { deleteProject, setProjectArchived, renameProject } = useProject()
  const { refreshAll } = useBranches()
  const { schedule } = useUndoDelete()
  const { run: archive, pending: archivePending, error: archiveError } = useAsyncAction(setProjectArchived, "Could not update project archive")
  const renameProjectTo = async (target: ProjectListItem, name: string) => {
    const slug = await renameProject(target.slug, name)
    if (slug !== target.slug) {
      onProjectRenamed?.(target.slug, slug)
      void refreshAll().catch(() => {})
    }
  }

  /** Archive or restore `slug` unless another archive change is running. */
  const changeArchive = (slug: string, archived: boolean) => {
    if (!archivePending) void archive(slug, archived)
  }

  const removeProject = (project: ProjectListItem) => schedule(`project:${project.slug}`, project.name, () => deleteProject(project.slug))

  return { archiveError, changeArchive, renameProjectTo, removeProject }
}
