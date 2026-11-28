"use client"

import { createContext, useCallback, useMemo, type ReactNode } from "react"
import {
  createBranch as apiCreateBranch,
  mergeBranch as apiMergeBranch,
  cascadeDelete as apiCascadeDelete,
  moveHistoryItem as apiMoveHistoryItem,
} from "@/lib/backend/api"
import { historiesResource } from "@/lib/backend/resources"
import type { BranchMeta } from "@/lib/domain/histories"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import { useResourceState } from "@/hooks/useResource"
import { refresh } from "@/lib/resource"

/** Chat id -> history filename. */
type ChatIdMap = Map<string, string>

/** The provider's actions, all stable: their own context, so a history refresh re-renders none of their consumers. */
interface BranchActions {
  onFileClick: (file: string, qaIndex?: number) => void
  refreshAll: () => Promise<void>
  createBranch: (parentFile: string, branchMessageIdx: number) => Promise<{ childFile: string; chatId: string }>
  mergeBranch: (childFile: string) => Promise<void>
  cascadeDelete: (filename: string) => Promise<string[]>
}

interface BranchContextValue extends BranchActions {
  branchMeta: Record<string, BranchMeta>
  chatIdMap: ChatIdMap
  visibleRootFiles: string[]
  historiesLoading: boolean
  moveHistory: (filename: string, targetDir: string) => Promise<void>
}

const BranchContext = createContext<BranchContextValue | null>(null)
const BranchActionsContext = createContext<BranchActions | null>(null)

const NO_META: Record<string, BranchMeta> = {}
const NO_FILES: string[] = []

const refreshAll = () => refresh(historiesResource)

/** Owns chat history list and branch metadata. */
export function BranchProvider({ onFileClick, onHistoriesMoved, children }: {
  onFileClick: (file: string, qaIndex?: number) => void
  onHistoriesMoved?: (moves: ReadonlyMap<string, string>) => void
  children: ReactNode
}) {
  const { value: histories, error } = useResourceState("Chat histories", historiesResource)
  const branchMeta = histories?.branchMeta ?? NO_META
  const rootFiles = histories?.files ?? NO_FILES
  const historiesLoading = histories === undefined && error === null

  const chatIdMap = useMemo(() => buildChatIdMap(branchMeta), [branchMeta])

  const visibleRootFiles = useMemo(
    () => rootFiles.filter((f) => !branchMeta[f]?.parent_id),
    [rootFiles, branchMeta],
  )
  const moveHistory = useCallback(async (filename: string, targetDir: string) => {
    const result = await apiMoveHistoryItem(filename, targetDir)
    await refreshAll()
    const pathsByChatId = buildChatIdMap(historiesResource.get()?.branchMeta ?? NO_META)
    const moves = new Map([[filename, result.new_path]])
    // The store carries descendant branches and may rename files on collision.
    for (const [oldPath, metadata] of Object.entries(branchMeta)) {
      const newPath = metadata.chat_id ? pathsByChatId.get(metadata.chat_id) : undefined
      if (newPath && newPath !== oldPath) moves.set(oldPath, newPath)
    }
    onHistoriesMoved?.(moves)
  }, [branchMeta, onHistoriesMoved])

  const doCreateBranch = useCallback(async (parentFile: string, branchMessageIdx: number) => {
    const result = await apiCreateBranch(parentFile, branchMessageIdx)
    await refreshAll()
    return { childFile: result.child_file, chatId: result.chat_id }
  }, [])

  const doMergeBranch = useCallback(async (childFile: string) => {
    await apiMergeBranch(childFile)
    await refreshAll()
  }, [])

  const doCascadeDelete = useCallback(async (filename: string) => {
    const result = await apiCascadeDelete(filename)
    await refreshAll()
    return result.deleted
  }, [])

  const actions = useMemo<BranchActions>(() => ({
    onFileClick,
    refreshAll,
    createBranch: doCreateBranch,
    mergeBranch: doMergeBranch,
    cascadeDelete: doCascadeDelete,
  }), [onFileClick, doCreateBranch, doMergeBranch, doCascadeDelete])
  const value = useMemo<BranchContextValue>(() => ({
    ...actions, branchMeta, chatIdMap, visibleRootFiles, historiesLoading, moveHistory,
  }), [actions, branchMeta, chatIdMap, visibleRootFiles, historiesLoading, moveHistory])

  return (
    <BranchActionsContext.Provider value={actions}>
      <BranchContext.Provider value={value}>
        {children}
      </BranchContext.Provider>
    </BranchActionsContext.Provider>
  )
}

export function useBranches() {
  return useRequiredContext(BranchContext, "useBranches must be inside BranchProvider")
}

export function useBranchActions() {
  return useRequiredContext(BranchActionsContext, "useBranchActions must be inside BranchProvider")
}

/** Index branch metadata by `chat_id` to resolve `parent_id` links back to filenames. */
function buildChatIdMap(meta: Record<string, BranchMeta>): ChatIdMap {
  const m = new Map<string, string>()
  for (const [k, v] of Object.entries(meta)) {
    if (v.chat_id) m.set(v.chat_id, k)
  }
  return m
}

/** Whether `file` is a strict ancestor of `activeFile` by walking `parent_id` links. Cycle-safe. */
export function isAncestorOf(activeFile: string | null | undefined, file: string, meta: Record<string, BranchMeta>, chatIdMap: ChatIdMap): boolean {
  if (!activeFile) return false
  const ancestorChatId = meta[file]?.chat_id
  if (!ancestorChatId) return false
  let current: string | null = activeFile
  const visited = new Set<string>()
  while (current && !visited.has(current)) {
    visited.add(current)
    const parentId: string | null | undefined = meta[current]?.parent_id
    if (parentId === ancestorChatId) return true
    if (!parentId) break
    current = chatIdMap.get(parentId) ?? null
  }
  return false
}