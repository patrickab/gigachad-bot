"use client"

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react"
import {
  createBranch as apiCreateBranch,
  mergeBranch as apiMergeBranch,
  cascadeDelete as apiCascadeDelete,
  fetchBranchMeta,
  listChatHistories,
  moveHistoryItem as apiMoveHistoryItem,
} from "@/lib/backend/api"
import type { BranchMeta } from "@/lib/domain/types"
import { useStartupRead } from "@/contexts/StartupReadContext"

/** Chat id -> history filename. */
type ChatIdMap = Map<string, string>

interface BranchContextValue {
  branchMeta: Record<string, BranchMeta>
  chatIdMap: ChatIdMap
  rootFiles: string[]
  visibleRootFiles: string[]
  historiesLoading: boolean
  activeFile: string | null
  activeQaIndex: number | null
  onFileClick: (file: string, qaIndex?: number) => void
  refreshAll: () => Promise<void>
  moveHistory: (filename: string, targetDir: string) => Promise<void>
  createBranch: (parentFile: string, branchMessageIdx: number) => Promise<{ childFile: string; chatId: string }>
  mergeBranch: (childFile: string) => Promise<void>
  cascadeDelete: (filename: string) => Promise<string[]>
  setActiveFile: (file: string | null) => void
  setActiveQaIndex: (idx: number | null) => void
}

const BranchContext = createContext<BranchContextValue | null>(null)

interface Histories {
  sortedFiles: string[]
  branchData: Record<string, BranchMeta>
}

/** Fetch both history sources before applying an explicit refresh. */
async function fetchHistories(): Promise<Histories> {
  const [histData, branchData] = await Promise.all([
    listChatHistories(),
    fetchBranchMeta(),
  ])
  const sortedFiles = (histData.files ?? []).slice().sort((a: string, b: string) => a.localeCompare(b))
  return { sortedFiles, branchData }
}

/** Owns chat history list, branch metadata and the active file/QA focus shown in the sidebar. */
export function BranchProvider({ onFileClick, onHistoriesMoved, children }: {
  onFileClick: (file: string, qaIndex?: number) => void
  onHistoriesMoved?: (moves: ReadonlyMap<string, string>) => void
  children: ReactNode
}) {
  const [branchMeta, setBranchMeta] = useState<Record<string, BranchMeta>>({})
  const [rootFiles, setRootFiles] = useState<string[]>([])
  const [activeFile, setActiveFile] = useState<string | null>(null)
  const [activeQaIndex, setActiveQaIndex] = useState<number | null>(null)

  const applyHistories = useCallback(({ sortedFiles, branchData }: Histories) => {
    setRootFiles(sortedFiles)
    setBranchMeta(branchData)
  }, [])

  const refreshAll = useCallback(async () => {
    applyHistories(await fetchHistories())
  }, [applyHistories])

  const { loading: historiesLoading } = useStartupRead("Chat histories", listChatHistories, (histData) => {
    setRootFiles((histData.files ?? []).slice().sort((a: string, b: string) => a.localeCompare(b)))
  })
  const loadBranchMeta = useCallback((signal: AbortSignal) => fetchBranchMeta(undefined, signal), [])
  useStartupRead("Branch metadata", loadBranchMeta, setBranchMeta)

  const chatIdMap = useMemo(() => buildChatIdMap(branchMeta), [branchMeta])

  const visibleRootFiles = useMemo(
    () => rootFiles.filter((f) => !branchMeta[f]?.parent_id),
    [rootFiles, branchMeta],
  )
  const moveHistory = useCallback(async (filename: string, targetDir: string) => {
    const result = await apiMoveHistoryItem(filename, targetDir)
    const histories = await fetchHistories()
    const pathsByChatId = buildChatIdMap(histories.branchData)
    const moves = new Map([[filename, result.new_path]])
    // The store carries descendant branches and may rename files on collision.
    for (const [oldPath, metadata] of Object.entries(branchMeta)) {
      const newPath = metadata.chat_id ? pathsByChatId.get(metadata.chat_id) : undefined
      if (newPath && newPath !== oldPath) moves.set(oldPath, newPath)
    }
    onHistoriesMoved?.(moves)
    setActiveFile((current) => current ? moves.get(current) ?? current : null)
    applyHistories(histories)
  }, [branchMeta, onHistoriesMoved, applyHistories])


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

  return (
    <BranchContext.Provider value={{
      branchMeta,
      chatIdMap,
      rootFiles,
      visibleRootFiles,
      historiesLoading,
      activeFile,
      activeQaIndex,
      onFileClick,
      refreshAll,
      moveHistory,
      createBranch: doCreateBranch,
      mergeBranch: doMergeBranch,
      cascadeDelete: doCascadeDelete,
      setActiveFile,
      setActiveQaIndex,
    }}>
      {children}
    </BranchContext.Provider>
  )
}

export function useBranches() {
  const ctx = useContext(BranchContext)
  if (!ctx) throw new Error("useBranches must be inside BranchProvider")
  return ctx
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