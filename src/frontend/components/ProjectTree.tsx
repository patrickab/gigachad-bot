"use client"

import { FileText, FolderKanban, LayoutDashboard } from "lucide-react"
import { VaultTree, type VaultTreeItem } from "./VaultTree"
import type { ProjectListItem } from "@/lib/types"

interface ProjectTreeProps {
  projects: ProjectListItem[]
  activeProject: string | null
  collapsed?: boolean
  open?: boolean
  onOpenChange?: (open: boolean) => void
  onExpand?: () => void
  onVaultClick?: (id: string) => void
  onVaultDelete?: (id: string) => void
  onAddVault?: (name: string) => Promise<void>
  onAddFolder?: (parentId: string | null, name: string) => Promise<void>
  onDashboardClick?: (vaultId: string) => void
  onElementClick?: (item: VaultTreeItem<string>) => void
}

export function ProjectTree({ projects, activeProject, ...rest }: ProjectTreeProps) {
  return (
    <VaultTree
      sectionIcon={FolderKanban}
      sectionTitle="Projects"
      storageKey="expanded_project_folders"
      count={projects.length}
      items={buildProjectItems(projects, activeProject)}
      {...rest}
    />
  )
}

function buildProjectItems(
  projects: ProjectListItem[],
  activeProject: string | null,
): VaultTreeItem<string>[] {
  return projects.map((project) => ({
    id: project.slug,
    label: project.name,
    type: "vault",
    data: project.slug,
    isActive: activeProject === project.slug,
    children: [
      {
        id: `${project.slug}/__dashboard__`,
        label: "Dashboard",
        type: "element",
        icon: LayoutDashboard,
        isSystem: true,
        data: project.slug,
      },
      ...(project.tabs ?? [])
        .filter((tab) => !tab.filename.startsWith("untitled-"))
        .map((tab) => ({
          id: `${project.slug}/${tab.filename}`,
          label: tab.name ?? tab.title ?? tab.filename.replace(".json", ""),
          type: "element" as const,
          data: `${project.slug}/${tab.filename}`,
          icon: FileText,
        })),
    ],
  }))
}
