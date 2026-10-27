"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { Factory } from "lucide-react"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { useTabActive, type WorkspaceItem } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { factoryItem } from "@/features/factory/items"
import { useFactoryRepos } from "@/features/factory/useFactoryRepos"
import { useOmpSidebar } from "@/features/omp/useOmpSidebar"
import { fileRootTree, removeFileRoot } from "@/lib/backend/api"
import type { FileTreeNode } from "@/lib/domain/files"
import { errorMessage } from "@/lib/utils"
import { useArchivedDirectories } from "./useArchivedDirectories"
import { useWorktrees } from "./useWorktrees"
import { mountOf } from "./workspaceTree"

interface CodeMountsOptions {
  onRootsChanged?: () => void
  onItemOpen?: (item: WorkspaceItem, inNewTab?: boolean) => void
}

/** File mounts of the Code section: their tree, worktrees, omp sessions and archive state. */
export function useCodeMounts({ onRootsChanged, onItemOpen }: CodeMountsOptions) {
  const hostAccess = useHostAccess()
  const isActive = useTabActive()
  const { schedule } = useUndoDelete()
  const { archived, setArchived } = useArchivedDirectories()
  const [rootTreeData, setRootTreeData] = useState<FileTreeNode[]>([])
  const mounts = rootTreeData.map(mountOf)
  const openFactoryRepo = useFactoryRepos(hostAccess)
  const omp = useOmpSidebar({
    onOpen: onItemOpen,
    workspaceActions: (directory, label, reportError) => !hostAccess ? [] : [{
      label: `Open factory for ${label}`,
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
  const { reload: reloadWorktrees } = worktrees
  const activeWorktreesKey = worktrees.activeWorktrees.join("\n")

  const worktreeKeys = useMemo(() => activeWorktreesKey.split("\n").filter(Boolean), [activeWorktreesKey])
  const readRootTree = useCallback((signal?: AbortSignal) => fileRootTree(worktreeKeys, signal), [worktreeKeys])
  const loadRootTree = useCallback(() => readRootTree().then((result) => setRootTreeData(result.tree)), [readRootTree])
  useStartupRead("Sidebar file mounts", readRootTree, (result) => setRootTreeData(result.tree), isActive)

  useEffect(() => {
    if (isActive) reloadWorktrees()
  }, [isActive, reloadWorktrees])

  /** Run `add`, then reload the tree and report the changed roots. */
  const mountThen = async (add: () => Promise<unknown>) => {
    await add()
    await loadRootTree()
    onRootsChanged?.()
  }

  const removeMount = (mount: string, name: string) => schedule(`root:${mount}`, name, async () => {
    await removeFileRoot(mount)
    await loadRootTree()
    onRootsChanged?.()
  })

  return {
    hostAccess, omp, rootTreeData, mounts, unmounted: omp.unmountedWorkspaces(mounts), folder: worktrees.folder,
    archived, setArchived, mountThen, removeMount,
  }
}
