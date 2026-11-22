"use client"

import { useEffect, useState } from "react"
import { Factory } from "lucide-react"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useTabActive, type WorkspaceItem } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { factoryItem } from "@/features/factory/items"
import { useFactoryRepos } from "@/features/factory/useFactoryRepos"
import { usePiRepositories } from "@/features/pi/sessionStore"
import { usePiSidebar } from "@/features/pi/usePiSidebar"
import { useResource } from "@/hooks/useResource"
import { addFileRoot, removeFileRoot } from "@/lib/backend/api"
import { fileRootsEnabledResource, refreshRootTree, revalidateOnActivation, rootTreeResource } from "@/lib/backend/resources"
import { mountOf, type FileTreeNode } from "@/lib/domain/files"
import { isInside, trimSlash } from "@/lib/paths"
import { errorMessage } from "@/lib/utils"
import { useArchivedDirectories } from "./useArchivedDirectories"
import { useWorktrees } from "./useWorktrees"

/** The tree before its first read. */
const NO_TREE: FileTreeNode[] = []

interface CodeMountsOptions {
  onItemOpen?: (item: WorkspaceItem, inNewTab?: boolean) => void
}

/** File mounts of the Code section: their tree, worktrees, Pi sessions and archive state. Every tab shares the tree and worktrees. */
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
  const sessions = usePiSidebar({
    onOpen: onItemOpen,
    workspaceActions: (directory, label, reportError) => !hostAccess ? [] : [{
      label: `Open factory for ${label}`,
      menuLabel: "Open factory",
      icon: Factory,
      inline: true,
      onClick: async () => {
        reportError("")
        try {
          onItemOpen?.(factoryItem((await openFactoryRepo(directory)).path))
        } catch (cause) {
          reportError(errorMessage(cause, "Could not open the factory"))
        }
      },
    }],
  })
  const repositories = usePiRepositories()
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

  const unmounted = sessions.unmountedWorkspaces(mounts)
  // Neither mounted nor holding a session: reachable only by mounting
  const unused = repositories.filter((repository) => !mounts.some((mount) => isInside(repository.path, mount)) && !unmounted.some((workspace) => workspace.path === trimSlash(repository.path)))
  const mountRepository = (path: string) => mountThen(() => addFileRoot(path))

  return {
    hostAccess, sessions, rootTreeData, mounts, unmounted, unused, mountRepository, folder: worktrees.folder,
    archived, setArchived, mountThen, removeMount,
  }
}
