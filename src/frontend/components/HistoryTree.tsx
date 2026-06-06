"use client"

import { Clock } from "lucide-react"
import { VaultTree, type VaultTreeItem } from "./VaultTree"
import type { ProjectListItem } from "@/lib/types"

interface HistoryTreeProps {
  rootFiles: string[]
  histories: Record<string, string[]>
  projects: ProjectListItem[]
  collapsed?: boolean
  open?: boolean
  onOpenChange?: (open: boolean) => void
  onExpand?: () => void
  onPlusClick?: () => void
  plusTitle?: string
  onElementClick?: (item: VaultTreeItem<string>) => void
  onElementArchive?: (item: VaultTreeItem<string>) => void
  onElementDelete?: (item: VaultTreeItem<string>) => void
  onVaultDelete?: (id: string) => void
  onAddFolder?: (parentId: string | null, name: string) => Promise<void>
  onMoveElement?: (elementId: string, targetId: string | null) => Promise<void>
}

export function HistoryTree({ rootFiles, histories, projects, ...rest }: HistoryTreeProps) {
  return (
    <VaultTree
      sectionIcon={Clock}
      sectionTitle="Histories"
      storageKey="expanded_history_folders"
      items={buildHistoryItems(rootFiles, histories, projects)}
      {...rest}
    />
  )
}

function buildHistoryItems(
  rootFiles: string[],
  histories: Record<string, string[]>,
  projects: ProjectListItem[],
): VaultTreeItem<string>[] {
  const projectSlugs = new Set(projects.map((p) => p.slug))

  const folderItems = Object.entries(histories)
    .filter(([dir]) => !projectSlugs.has(dir))
    .map(([dir, files]) => ({
      id: dir,
      label: dir,
      type: "folder" as const,
      data: dir,
      children: files.map((file) => ({
        id: `${dir}/${file}`,
        label: file.replace(".json", ""),
        type: "element" as const,
        data: `${dir}/${file}`,
      })),
    }))

  const rootItems = rootFiles.map((file) => ({
    id: file,
    label: file.replace(".json", ""),
    type: "element" as const,
    data: file,
  }))

  return [...folderItems, ...rootItems]
}
