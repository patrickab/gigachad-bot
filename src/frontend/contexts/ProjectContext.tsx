"use client"

import { createContext, useCallback, useEffect, useMemo, useState, type ReactNode } from "react"
import type { ProjectListItem } from "@/lib/domain/projects"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import { subscribeToChanges } from "@/lib/backend/syncStream"
import {
  listProjects as apiListProjects,
  createProject as apiCreateProject,
  deleteProject as apiDeleteProject,
  setProjectArchived as apiSetProjectArchived,
  renameProject as apiRenameProject,
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

/** Owns the project list. Which project a chat belongs to is the tab's `projectSlug`, never provider state. */
export function ProjectProvider({ children }: { children: ReactNode }) {
  const [projects, setProjects] = useState<ProjectListItem[]>([])
  const refreshProjects = useCallback(async () => {
    setProjects(await apiListProjects())
  }, [])

  // Other devices' creates, renames and deletes all rewrite the catalog.
  useEffect(() => subscribeToChanges((event) => {
    if (event.resource_kind === "document" && event.resource_key === PROJECT_CATALOG_KEY) void refreshProjects().catch(() => {})
  }), [refreshProjects])

  useStartupRead("Projects", apiListProjects, setProjects)

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
