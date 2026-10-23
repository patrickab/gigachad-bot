"use client"

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react"
import type { ProjectListItem } from "@/lib/domain/types"
import { useStartupRead } from "@/contexts/StartupReadContext"
import {
  listProjects as apiListProjects,
  createProject as apiCreateProject,
  deleteProject as apiDeleteProject,
} from "@/lib/backend/api"

interface ProjectContextState {
  projects: ProjectListItem[]
  refreshProjects: () => Promise<void>
  createProject: (name: string) => Promise<string>
  deleteProject: (name: string) => Promise<void>
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

  const value = useMemo<ProjectContextState>(
    () => ({ projects, refreshProjects, createProject, deleteProject }),
    [projects, refreshProjects, createProject, deleteProject],
  )

  return (
    <ProjectContext.Provider value={value}>
      {children}
    </ProjectContext.Provider>
  )
}
