"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import {
  createBranch as apiCreateBranch,
  mergeBranch as apiMergeBranch,
  cascadeDelete as apiCascadeDelete,
  orphanChildren as apiOrphanChildren,
  fetchBranchMeta,
  listChatHistories,
} from "@/lib/api"
import type { BranchMeta } from "@/lib/types"

export interface UseBranchesReturn {
  branchMeta: Record<string, BranchMeta>
  rootFiles: string[]
  histories: Record<string, string[]>
  visibleRootFiles: string[]
  visibleHistories: Record<string, string[]>
  historiesLoading: boolean
  refreshAll: () => Promise<void>
  createBranch: (parentFile: string, branchMessageIdx: number) => Promise<{ childFile: string; chatId: string }>
  mergeBranch: (childFile: string) => Promise<void>
  cascadeDelete: (filename: string) => Promise<string[]>
  orphanChildren: (filename: string) => Promise<string[]>
}

export function useBranches(): UseBranchesReturn {
  const [branchMeta, setBranchMeta] = useState<Record<string, BranchMeta>>({})
  const [rootFiles, setRootFiles] = useState<string[]>([])
  const [histories, setHistories] = useState<Record<string, string[]>>({})
  const [historiesLoading, setHistoriesLoading] = useState(true)

  const refreshAll = useCallback(async () => {
    const [histData, branchData] = await Promise.all([
      listChatHistories().catch(() => ({ files: [] as string[], histories: {} as Record<string, string[]> })),
      fetchBranchMeta().catch(() => ({} as Record<string, BranchMeta>)),
    ])
    const sortedFiles = (histData.files ?? []).slice().sort((a: string, b: string) => a.localeCompare(b))
    const rawHistories = histData.histories ?? {}
    const sortedHistories: Record<string, string[]> = {}
    for (const dir of Object.keys(rawHistories).sort((a: string, b: string) => a.localeCompare(b))) {
      sortedHistories[dir] = rawHistories[dir].slice().sort((a: string, b: string) => a.localeCompare(b))
    }
    setRootFiles(sortedFiles)
    setHistories(sortedHistories)
    setBranchMeta(branchData)
    setHistoriesLoading(false)
  }, [])

  useEffect(() => {
    refreshAll()
  }, [refreshAll])

  const visibleRootFiles = useMemo(
    () => rootFiles.filter((f) => !branchMeta[f]?.parent_id),
    [rootFiles, branchMeta],
  )

  const visibleHistories = useMemo(() => {
    const filtered: Record<string, string[]> = {}
    for (const [dir, files] of Object.entries(histories)) {
      filtered[dir] = files.filter((f) => {
        const key = `${dir}/${f}`
        return !branchMeta[key]?.parent_id
      })
    }
    return filtered
  }, [histories, branchMeta])

  const doCreateBranch = useCallback(async (parentFile: string, branchMessageIdx: number) => {
    const result = await apiCreateBranch(parentFile, branchMessageIdx)
    await refreshAll()
    return { childFile: result.child_file, chatId: result.chat_id }
  }, [refreshAll])

  const doMergeBranch = useCallback(async (childFile: string) => {
    await apiMergeBranch(childFile)
    await refreshAll()
  }, [refreshAll])

  const doCascadeDelete = useCallback(async (filename: string) => {
    const result = await apiCascadeDelete(filename)
    await refreshAll()
    return result.deleted
  }, [refreshAll])

  const doOrphanChildren = useCallback(async (filename: string) => {
    const result = await apiOrphanChildren(filename)
    await refreshAll()
    return result.orphaned
  }, [refreshAll])

  return {
    branchMeta,
    rootFiles,
    histories,
    visibleRootFiles,
    visibleHistories,
    historiesLoading,
    refreshAll,
    createBranch: doCreateBranch,
    mergeBranch: doMergeBranch,
    cascadeDelete: doCascadeDelete,
    orphanChildren: doOrphanChildren,
  }
}