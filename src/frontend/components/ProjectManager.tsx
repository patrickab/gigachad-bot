"use client"

import { FolderKanban, Loader2, Plus, Trash2, X, LayoutDashboard } from "lucide-react"
import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { useProject } from "@/contexts/ProjectContext"
import { useSafeAction } from "@/hooks/useSafeAction"
import { SidebarElement } from "./SidebarElement"
import { Skeleton } from "./Skeleton"

interface ProjectManagerProps {
  collapsed?: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  onExpand?: () => void
}

export function ProjectManager({ collapsed, open, onOpenChange, onExpand }: ProjectManagerProps) {
  const { projects, projectsLoading, activeProject, openProject, createProject, deleteProject, setDashboardOpen } = useProject()
  const { run, error } = useSafeAction()
  const sortedProjects = [...projects].sort((a, b) => a.name.localeCompare(b.name))
  const [newProjectName, setNewProjectName] = useState("")
  const [showNewProject, setShowNewProject] = useState(false)

  const handleCreateProject = () => run("create project", async () => {
    if (!newProjectName.trim()) return
    await createProject(newProjectName.trim())
    setNewProjectName("")
    setShowNewProject(false)
  })

  const handleOpenProject = (slug: string) => run("open project", () => openProject(slug))
  const handleDeleteProject = (slug: string) => run("delete project", () => deleteProject(slug))

  return (
    <div className="space-y-1 w-full">
      {collapsed ? (
        <SidebarElement
          icon={FolderKanban}
          title="Projects"
          collapsed={true}
          onClick={() => {
            if (onExpand) onExpand()
            onOpenChange(true)
          }}
        />
      ) : (
        <>
          <div className="flex w-full items-center justify-between p-1.5 rounded-md text-zinc-400 hover:bg-zinc-800/20 group">
            <button
              onClick={() => onOpenChange(!open)}
              className="flex-1 flex items-center gap-3 p-1 rounded-md text-left transition-colors hover:text-zinc-200"
            >
              <FolderKanban className="h-4 w-4 shrink-0 text-zinc-400" />
              <span className="text-sm font-medium">Projects</span>
              {projects.length > 0 && (
                <span className="rounded-full bg-zinc-850 px-1.5 py-0.5 text-[10px] text-zinc-500 font-semibold">{projects.length}</span>
              )}
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation()
                onOpenChange(true)
                setShowNewProject(true)
              }}
              className="p-1 rounded text-zinc-500 hover:text-cyan-400 hover:bg-zinc-900 transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
              title="New project"
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>

          <AnimatePresence>
            {open && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="space-y-0.5 pl-1">
                  {projectsLoading ? (
                    <div className="space-y-1.5 px-2 py-1">
                      <Skeleton className="h-3 w-20" />
                      <Skeleton className="h-3 w-28" />
                    </div>
                  ) : (
                    <>
                      {sortedProjects.map((proj) => {
                        const isActive = activeProject === proj.slug
                        return (
                          <div key={proj.slug} className="flex flex-col gap-0.5">
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              className={cn(
                                "flex items-center gap-2 pl-4 pr-2 py-1 rounded-md transition-all duration-200 ease-out border border-transparent",
                                isActive
                                  ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
                                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/40"
                              )}
                            >
                              <button
                                onClick={() => handleOpenProject(proj.slug)}
                                className={cn(
                                  "flex-1 truncate text-left text-xs transition-colors",
                                  isActive ? "font-semibold text-cyan-400" : "font-medium"
                                )}
                              >
                                {proj.name}
                              </button>
                              <button
                                onClick={(e) => { e.stopPropagation(); handleDeleteProject(proj.slug) }}
                                className="rounded p-0.5 text-zinc-600 hover:text-red-400 transition-colors"
                              >
                                <Trash2 className="h-3 w-3" />
                              </button>
                            </motion.div>
                            {isActive && (
                              <motion.div
                                initial={{ opacity: 0, y: -4 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -4 }}
                                className="pl-8 pr-2 py-1"
                              >
                                <button
                                  onClick={() => setDashboardOpen(true)}
                                  className="flex items-center gap-2 text-[11px] text-zinc-500 hover:text-cyan-400 transition-colors w-full text-left font-medium"
                                >
                                  <LayoutDashboard className="h-3.5 w-3.5 shrink-0" />
                                  <span>Dashboard</span>
                                </button>
                              </motion.div>
                            )}
                          </div>
                        )
                      })}
                      {showNewProject && (
                        <div className="flex items-center gap-1 pl-5 pr-2 py-0.5">
                          <input
                            autoFocus
                            value={newProjectName}
                            onChange={(e) => setNewProjectName(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleCreateProject()
                              if (e.key === "Escape") { setShowNewProject(false); setNewProjectName("") }
                            }}
                            className="flex-1 bg-transparent border border-zinc-700 rounded px-1.5 py-0.5 text-[11px] text-zinc-200 outline-none focus:border-cyan-500/40"
                            placeholder="Project name"
                          />
                          <button
                            onClick={handleCreateProject}
                            className="text-[10px] text-cyan-400 hover:text-cyan-300"
                          >
                            Create
                          </button>
                          <button
                            onClick={() => { setShowNewProject(false); setNewProjectName("") }}
                            className="text-zinc-600 hover:text-zinc-300"
                          >
                            <X className="h-3 w-3" />
                          </button>
                        </div>
                      )}
                      {projects.length === 0 && !showNewProject && (
                        <p className="px-2 py-1 text-[11px] text-zinc-600">No projects yet.</p>
                      )}
                    </>
                  )}
                  {error && (
                    <p className="px-2 py-1 text-[11px] text-red-400">{error}</p>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </div>
  )
}