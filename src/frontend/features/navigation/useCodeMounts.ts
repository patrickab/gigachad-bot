"use client"

import { useEffect, useState } from "react"
import { Factory } from "lucide-react"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useTabActive, type WorkspaceItem } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { factoryItem } from "@/features/factory/items"
import { useFactoryRepos } from "@/features/factory/useFactoryRepos"
import { useOmpSidebar } from "@/features/omp/useOmpSidebar"
import { useResource } from "@/hooks/useResource"
import { removeFileRoot } from "@/lib/backend/api"
import { fileRootsEnabledResource, refreshRootTree, revalidateOnActivation, rootTreeResource } from "@/lib/backend/resources"
import { mountOf, type FileTreeNode } from "@/lib/domain/files"
import { errorMessage } from "@/lib/utils"
import { useArchivedDirectories } from "./useArchivedDirectories"
import { useWorktrees } from "./useWorktrees"

/** The tree before its first read. */
const NO_TREE: FileTreeNode[] = []

interface CodeMountsOptions {
  onItemOpen?: (item: WorkspaceItem, inNewTab?: boolean) => void
}

/** File mounts of the Code section: their tree, worktrees, omp sessions and archive state. Every tab shares the tree and worktrees. */
export function useCodeMounts({ onItemOpen }: CodeMountsOptions) {
  const hostAccess = useHostAccess()
  const isActive = useTabActive()
  const { schedule } = useUndoDelete()
  const { archived, setArchived } = useArchivedDirectories()
  // The tree read for other worktrees replaces the shown one once it arrives, so picking a worktree never empties the section.
  const [shownTree, setShownTree] = useState<FileTreeNode[]>()
  const rootTreeData = shownTree ?? NO_TREE
  const mounts = rootTreeData.map(mountOf)
  const openFactoryRepo = useFactoryRepos(hostAccess)
  const omp = useOmpSidebar({
    onOpen: onItemOpen,
    workspaceActions: (directory, label, reportError) => !hostAccess ? [] : [{
      label: `Open factory for ${label}`,
      menuLabel: "Open factory",
      icon: Factory,
      onClick: async () => {
        reportError("")
        try {
          onItemOpen?.(factoryItem((await openFactoryRepo(directory)).name))
        } catch (cause) {
          reportError(errorMessage(cause, "Could not open the factory"))
        }
      },
    }],
  })
  const worktrees = useWorktrees(mounts)
  const rootTree = rootTreeResource(worktrees.activeWorktrees.join("\n"))
  const loadedTree = useResource("Sidebar file mounts", rootTree)
  if (loadedTree !== undefined && loadedTree !== shownTree) setShownTree(loadedTree)

  // Nothing announces changes on disk or to a worktree's status: the tab coming to the front reads them again for every tab,
  // and so does showing other worktrees, whose tree may be one read long before.
  useEffect(() => {
    if (isActive) revalidateOnActivation()
  }, [isActive, rootTree])

  /** Run `add`, then read the tree again; the file picker's availability may change with it. */
  const mountThen = async (add: () => Promise<unknown>) => {
    await add()
    await refreshRootTree()
    fileRootsEnabledResource.invalidate()
  }

  const removeMount = (mount: string, name: string) => schedule(`root:${mount}`, name, async () => {
    await removeFileRoot(mount)
    await refreshRootTree()
    fileRootsEnabledResource.invalidate()
  })

  return {
    hostAccess, omp, rootTreeData, mounts, unmounted: omp.unmountedWorkspaces(mounts), folder: worktrees.folder,
    archived, setArchived, mountThen, removeMount,
  }
}
