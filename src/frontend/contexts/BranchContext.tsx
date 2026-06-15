"use client"

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import {
  createBranch as apiCreateBranch,
  mergeBranch as apiMergeBranch,
  cascadeDelete as apiCascadeDelete,
  orphanChildren as apiOrphanChildren,
  fetchBranchMeta,
  listChatHistories,
} from "@/lib/api"
import type { BranchMeta } from "@/lib/types"

type ChatIdMap = Map<string, string>

export interface BranchContextValue {
  branchMeta: Record<string, BranchMeta>
  chatIdMap: ChatIdMap
  rootFiles: string[]
  histories: Record<string, string[]>
  visibleRootFiles: string[]
  visibleHistories: Record<string, string[]>
  historiesLoading: boolean
  activeFile: string | null
  activeQaIndex: number | null
  onFileClick: (file: string, qaIndex?: number) => void
  onMerge: (childFile: string) => Promise<void>
  onDelete: (file: string) => void
  refreshAll: () => Promise<void>
  createBranch: (parentFile: string, branchMessageIdx: number) => Promise<{ childFile: string; chatId: string }>
  mergeBranch: (childFile: string) => Promise<void>
  cascadeDelete: (filename: string) => Promise<string[]>
  orphanChildren: (filename: string) => Promise<string[]>
  setActiveFile: (file: string | null) => void
  setActiveQaIndex: (idx: number | null) => void
  registerOnFileClick: (fn: (file: string, qaIndex?: number) => void) => void
  registerOnMerge: (fn: (childFile: string) => Promise<void>) => void
  registerOnDelete: (fn: (file: string) => Promise<void>) => void
}

const BranchContext = createContext<BranchContextValue | null>(null)

export function BranchProvider({ children }: { children: ReactNode }) {
  const [branchMeta, setBranchMeta] = useState<Record<string, BranchMeta>>({})
  const [rootFiles, setRootFiles] = useState<string[]>([])
  const [histories, setHistories] = useState<Record<string, string[]>>({})
  const [historiesLoading, setHistoriesLoading] = useState(true)
  const [activeFile, setActiveFile] = useState<string | null>(null)
  const [activeQaIndex, setActiveQaIndex] = useState<number | null>(null)

  const onFileClickRef = useRef<(file: string, qaIndex?: number) => void>(() => {})
  const onMergeRef = useRef<(childFile: string) => Promise<void>>(async () => {})
  const onDeleteRef = useRef<(file: string) => Promise<void>>(async () => {})

  const registerOnFileClick = useCallback((fn: (file: string, qaIndex?: number) => void) => {
    onFileClickRef.current = fn
  }, [])

  const registerOnMerge = useCallback((fn: (childFile: string) => Promise<void>) => {
    onMergeRef.current = fn
  }, [])

  const registerOnDelete = useCallback((fn: (file: string) => Promise<void>) => {
    onDeleteRef.current = fn
  }, [])

  const onFileClick = useCallback((file: string, qaIndex?: number) => {
    onFileClickRef.current(file, qaIndex)
  }, [])

  const onMerge = useCallback((childFile: string) => {
    return onMergeRef.current(childFile)
  }, [])

  const onDelete = useCallback((file: string) => {
    void onDeleteRef.current(file)
  }, [])

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

  const chatIdMap = useMemo(() => buildChatIdMap(branchMeta), [branchMeta])

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

  return (
    <BranchContext.Provider value={{
      branchMeta,
      chatIdMap,
      rootFiles,
      histories,
      visibleRootFiles,
      visibleHistories,
      historiesLoading,
      activeFile,
      activeQaIndex,
      onFileClick,
      onMerge,
      onDelete,
      refreshAll,
      createBranch: doCreateBranch,
      mergeBranch: doMergeBranch,
      cascadeDelete: doCascadeDelete,
      orphanChildren: doOrphanChildren,
      setActiveFile,
      setActiveQaIndex,
      registerOnFileClick,
      registerOnMerge,
      registerOnDelete,
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

export function useBranchMeta() {
  const ctx = useContext(BranchContext)
  return ctx
}

export function useBranchContext() {
  const ctx = useContext(BranchContext)
  if (!ctx) throw new Error("useBranchContext must be inside BranchProvider")
  return ctx
}

export function buildChatIdMap(meta: Record<string, BranchMeta>): ChatIdMap {
  const m = new Map<string, string>()
  for (const [k, v] of Object.entries(meta)) {
    if (v.chat_id) m.set(v.chat_id, k)
  }
  return m
}

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