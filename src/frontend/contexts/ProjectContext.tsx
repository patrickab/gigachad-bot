"use client"

import { createContext, useCallback, useContext, useMemo, type ReactNode } from "react"
import { displayName, type ProjectListItem } from "@/lib/domain/projects"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import { useResource } from "@/hooks/useResource"
import { announceFileChange, announceFileMove } from "@/lib/backend/fileEvents"
import { forgetProjectDocuments, projectsResource } from "@/lib/backend/resources"
import { refresh } from "@/lib/resource"
import {
  createProject as apiCreateProject,
  deleteProject as apiDeleteProject,
  setProjectArchived as apiSetProjectArchived,
  renameProject as apiRenameProject,
  renameDocument as apiRenameDocument,
  type RenamedDocument,
} from "@/lib/backend/api"

interface ProjectContextState {
  projects: ProjectListItem[]
  refreshProjects: () => Promise<void>
  createProject: (name: string) => Promise<string>
  deleteProject: (name: string) => Promise<void>
  setProjectArchived: (slug: string, archived: boolean) => Promise<void>
  renameProject: (slug: string, name: string) => Promise<string>
}

const ProjectContext = createContext<ProjectContextState | null>(null)

export function useProject(): ProjectContextState {
  return useRequiredContext(ProjectContext, "useProject must be used within ProjectProvider")
}

const NO_PROJECTS: ProjectListItem[] = []

/** `(slug, path)` to the display name project `slug` gives its file `path`; none outside a ProjectProvider. */
export function useDisplayName(): (slug: string | null | undefined, path: string) => string | undefined {
  const projects = useContext(ProjectContext)?.projects ?? NO_PROJECTS
  return useCallback((slug, path) => displayName(projects, slug, path), [projects])
}

/** Rename `path` of project `slug` (a note when empty), read the projects again so every view shows the new name, and announce a moved file and the files rewritten to follow it. */
export function useRenameDocument(): (slug: string, path: string, name: string) => Promise<RenamedDocument> {
  const refreshProjects = useContext(ProjectContext)?.refreshProjects
  return useCallback(async (slug, path, name) => {
    const renamed = await apiRenameDocument(slug, path, name)
    await refreshProjects?.().catch(() => {})
    if (renamed.path !== path) {
      announceFileMove({ from: path, to: renamed.path })
      for (const changed of renamed.changed) announceFileChange(changed)
    }
    return renamed
  }, [refreshProjects])
}

/** Binds the project mutations to the shared project list. Which project a chat belongs to is the tab's `projectSlug`, never provider state. */
export function ProjectProvider({ children }: { children: ReactNode }) {
  const projects = useResource("Projects", projectsResource) ?? NO_PROJECTS
  const refreshProjects = useCallback(() => refresh(projectsResource), [])

  const thenRefresh = useCallback(async <T,>(call: Promise<T>) => {
    const result = await call
    await refreshProjects()
    return result
  }, [refreshProjects])

  const createProject = useCallback(async (name: string) => (await thenRefresh(apiCreateProject(name))).slug, [thenRefresh])
  const deleteProject = useCallback(async (slug: string) => {
    await apiDeleteProject(slug)
    forgetProjectDocuments(slug)
    await refreshProjects()
  }, [refreshProjects])
  const setProjectArchived = useCallback((slug: string, archived: boolean) => thenRefresh(apiSetProjectArchived(slug, archived)), [thenRefresh])
  const renameProject = useCallback(async (slug: string, name: string) => {
    const renamed = (await apiRenameProject(slug, name)).slug
    if (renamed !== slug) forgetProjectDocuments(slug)
    await refreshProjects()
    return renamed
  }, [refreshProjects])

  const value = useMemo<ProjectContextState>(
    () => ({ projects, refreshProjects, createProject, deleteProject, setProjectArchived, renameProject }),
    [projects, refreshProjects, createProject, deleteProject, setProjectArchived, renameProject],
  )

  return (
    <ProjectContext.Provider value={value}>
      {children}
    </ProjectContext.Provider>
  )
}
