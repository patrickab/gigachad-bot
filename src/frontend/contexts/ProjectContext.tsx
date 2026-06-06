"use client"

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react"
import type { ProjectData, ProjectListItem, ProjectStateUpdate, ProjectTab } from "@/lib/types"
import {
  listProjects as apiListProjects,
  loadProject as apiLoadProject,
  createProject as apiCreateProject,
  deleteProject as apiDeleteProject,
  addProjectCard as apiAddProjectCard,
  moveProjectCard as apiMoveProjectCard,
  deleteProjectCard as apiDeleteProjectCard,
  updateProjectKanban as apiUpdateProjectKanban,
} from "@/lib/api"

export interface ProjectContextState {
  activeProject: string | null
  projectData: ProjectData | null
  projects: ProjectListItem[]
  projectsLoading: boolean
  dashboardOpen: boolean
  openProject: (name: string) => Promise<void>
  closeProject: () => void
  createProject: (name: string) => Promise<void>
  deleteProject: (name: string) => Promise<void>
  refreshProjects: () => Promise<void>
  addCard: (title: string, description: string, state?: string) => Promise<void>
  moveCard: (cardId: string, state: string) => Promise<void>
  deleteCard: (cardId: string) => Promise<void>
  updateKanban: (data: ProjectData) => Promise<void>
  syncTabs: (tabs: { id: string; name: string | null; historyFile: string | null; title: string | null; chatId: string }[], updateHistoryFile: (tabId: string, historyFile: string) => void) => Promise<void>
  setDashboardOpen: (open: boolean) => void
}

const ProjectContext = createContext<ProjectContextState | null>(null)

export function useProject(): ProjectContextState {
  const ctx = useContext(ProjectContext)
  if (!ctx) throw new Error("useProject must be used within ProjectProvider")
  return ctx
}

export function ProjectProvider({ children }: { children: ReactNode }) {
  const [activeProject, setActiveProject] = useState<string | null>(null)
  const [projectData, setProjectData] = useState<ProjectData | null>(null)
  const [projects, setProjects] = useState<ProjectListItem[]>([])
  const [projectsLoading, setProjectsLoading] = useState(false)
  const [dashboardOpen, setDashboardOpen] = useState(false)

  // Single monotonic counter bumped on any project-switch event. Async calls
  // capture the current value and drop their result if the counter advanced
  // (i.e. the user switched project mid-flight, or closed/deleted it).
  const syncVersionRef = useRef(0)
  const bumpVersion = useCallback(() => { syncVersionRef.current++ }, [])

  const runGuarded = useCallback(
    async <T,>(fn: (project: string) => Promise<T>): Promise<T | null> => {
      if (!activeProject) return null
      const version = syncVersionRef.current
      const project = activeProject
      const result = await fn(project)
      return syncVersionRef.current === version ? result : null
    },
    [activeProject],
  )

  const refreshProjects = useCallback(async () => {
    setProjectsLoading(true)
    try {
      const list = await apiListProjects()
      setProjects(list)
    } finally {
      setProjectsLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshProjects()
  }, [refreshProjects])

  const openProject = useCallback(async (name: string) => {
    bumpVersion()
    const version = syncVersionRef.current
    const data = await apiLoadProject(name)
    if (syncVersionRef.current !== version) return
    setActiveProject(name)
    setProjectData(data)
  }, [bumpVersion])

  const closeProject = useCallback(() => {
    bumpVersion()
    setActiveProject(null)
    setProjectData(null)
  }, [bumpVersion])

  const createProject = useCallback(async (name: string) => {
    await apiCreateProject(name)
    await refreshProjects()
  }, [refreshProjects])

  const deleteProject = useCallback(async (name: string) => {
    await apiDeleteProject(name)
    if (activeProject === name) {
      bumpVersion()
      setActiveProject(null)
      setProjectData(null)
    }
    await refreshProjects()
  }, [activeProject, refreshProjects, bumpVersion])

  const addCard = useCallback(async (title: string, description: string, state: string = "backlog") => {
    const card = await runGuarded((p) => apiAddProjectCard(p, title, description, state))
    if (card) setProjectData((prev) => prev ? { ...prev, kanban: [...prev.kanban, card] } : prev)
  }, [runGuarded])

  const moveCard = useCallback(async (cardId: string, state: string) => {
    const updated = await runGuarded((p) => apiMoveProjectCard(p, cardId, state))
    if (updated) setProjectData((prev) => prev ? {
      ...prev,
      kanban: prev.kanban.map((c) => c.id === cardId ? updated : c),
    } : prev)
  }, [runGuarded])

  const deleteCard = useCallback(async (cardId: string) => {
    const ok = await runGuarded((p) => apiDeleteProjectCard(p, cardId).then(() => true))
    if (ok) setProjectData((prev) => prev ? {
      ...prev,
      kanban: prev.kanban.filter((c) => c.id !== cardId),
    } : prev)
  }, [runGuarded])

  const updateKanban = useCallback(async (data: ProjectData) => {
    const updated = await runGuarded((p) => apiUpdateProjectKanban(p, data))
    if (updated) setProjectData(updated)
  }, [runGuarded])

  const syncTabs = useCallback(async (
    tabs: { id: string; name: string | null; historyFile: string | null; title: string | null; chatId: string }[],
    updateHistoryFile: (tabId: string, historyFile: string) => void,
  ) => {
    if (!activeProject || !projectData) return
    const version = syncVersionRef.current
    const projectAtCall = activeProject
    const projectTabs: ProjectTab[] = []
    for (const t of tabs) {
      if (t.historyFile) {
        const filename = t.historyFile.includes("/") ? t.historyFile.split("/").slice(1).join("/") : t.historyFile
        projectTabs.push({ filename, name: t.name, title: t.title })
      } else {
        const shortId = t.chatId.replace(/[^a-zA-Z0-9]/g, "").slice(0, 12).toLowerCase() || "tab"
        const filename = `untitled-${shortId}.json`
        projectTabs.push({ filename, name: t.name, title: t.title })
        updateHistoryFile(t.id, `${projectAtCall}/${filename}`)
      }
    }
    const stateUpdate: ProjectStateUpdate = { kanban: projectData.kanban, tabs: projectTabs }
    const updated = await apiUpdateProjectKanban(projectAtCall, stateUpdate)
    if (syncVersionRef.current !== version) return
    setProjectData(updated)
  }, [activeProject, projectData])

  const value: ProjectContextState = {
    activeProject,
    projectData,
    projects,
    projectsLoading,
    dashboardOpen,
    openProject,
    closeProject,
    createProject,
    deleteProject,
    refreshProjects,
    addCard,
    moveCard,
    deleteCard,
    updateKanban,
    syncTabs,
    setDashboardOpen,
  }

  return (
    <ProjectContext.Provider value={value}>
      {children}
    </ProjectContext.Provider>
  )
}
