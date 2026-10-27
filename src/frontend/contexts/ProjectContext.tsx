"use client"

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react"
import { displayName, type ProjectDocument, type ProjectListItem } from "@/lib/domain/projects"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import { subscribeToChanges } from "@/lib/backend/syncStream"
import {
  listProjects as apiListProjects,
  createProject as apiCreateProject,
  deleteProject as apiDeleteProject,
  setProjectArchived as apiSetProjectArchived,
  renameProject as apiRenameProject,
  renameDocument as apiRenameDocument,
} from "@/lib/backend/api"

const PROJECT_CATALOG_KEY = "chat/projects-meta.json"

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

/** Rename `path` of project `slug` (a note when empty), then read the projects again so every view shows the new name. */
export function useRenameDocument(): (slug: string, path: string, name: string) => Promise<ProjectDocument> {
  const refreshProjects = useContext(ProjectContext)?.refreshProjects
  return useCallback(async (slug, path, name) => {
    const renamed = await apiRenameDocument(slug, path, name)
    await refreshProjects?.().catch(() => {})
    return renamed
  }, [refreshProjects])
}

/** Whether two project lists hold the same entries, so a refresh that changed nothing keeps the list's identity. */
function sameProjects(current: ProjectListItem[], next: ProjectListItem[]): boolean {
  return current.length === next.length && JSON.stringify(current) === JSON.stringify(next)
}

/** Owns the project list. Which project a chat belongs to is the tab's `projectSlug`, never provider state. */
export function ProjectProvider({ children }: { children: ReactNode }) {
  const [projects, setProjects] = useState<ProjectListItem[]>([])
  // Startup refreshes (change-feed replay, the first branch metadata) usually return the list unchanged.
  // Keeping its identity stops consumers keyed on it, e.g. the sidebar's per-project document reads, from reading again.
  const applyProjects = useCallback((next: ProjectListItem[]) => {
    setProjects((current) => sameProjects(current, next) ? current : next)
  }, [])
  const refreshProjects = useCallback(async () => {
    applyProjects(await apiListProjects())
  }, [applyProjects])

  // Other devices' creates, renames and deletes all rewrite the catalog.
  useEffect(() => subscribeToChanges((event) => {
    if (event.resource_kind === "document" && event.resource_key === PROJECT_CATALOG_KEY) void refreshProjects().catch(() => {})
  }), [refreshProjects])

  useStartupRead("Projects", apiListProjects, applyProjects)

  const thenRefresh = useCallback(async <T,>(call: Promise<T>) => {
    const result = await call
    await refreshProjects()
    return result
  }, [refreshProjects])

  const createProject = useCallback(async (name: string) => (await thenRefresh(apiCreateProject(name))).slug, [thenRefresh])
  const deleteProject = useCallback((name: string) => thenRefresh(apiDeleteProject(name)), [thenRefresh])
  const setProjectArchived = useCallback((slug: string, archived: boolean) => thenRefresh(apiSetProjectArchived(slug, archived)), [thenRefresh])
  const renameProject = useCallback(async (slug: string, name: string) => (await thenRefresh(apiRenameProject(slug, name))).slug, [thenRefresh])

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
