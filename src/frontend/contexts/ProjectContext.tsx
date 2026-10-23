"use client"

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react"
import type { ProjectListItem } from "@/lib/domain/types"
import { useStartupRead } from "@/contexts/StartupReadContext"
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
  const ctx = useContext(ProjectContext)
  if (!ctx) throw new Error("useProject must be used within ProjectProvider")
  return ctx
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

  const createProject = useCallback(async (name: string) => {
    const created = await apiCreateProject(name)
    await refreshProjects()
    return created.slug
  }, [refreshProjects])

  const deleteProject = useCallback(async (name: string) => {
    await apiDeleteProject(name)
    await refreshProjects()
  }, [refreshProjects])

  const setProjectArchived = useCallback(async (slug: string, archived: boolean) => {
    await apiSetProjectArchived(slug, archived)
    await refreshProjects()
  }, [refreshProjects])

  const renameProject = useCallback(async (slug: string, name: string) => {
    const renamed = await apiRenameProject(slug, name)
    await refreshProjects()
    return renamed.slug
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
