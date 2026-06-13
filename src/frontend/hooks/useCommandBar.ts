"use client"

import { useCallback, useRef, useState } from "react"
import { cancelMemories, commitMemoryDocs, composeMemoryDocs, extractMemories } from "@/lib/api"
import type { MemoryComposeResponse, MemoryExtractResponse, ProposedMemory } from "@/lib/types"

export type CommandBarPhase = "idle" | "input" | "extracting" | "review" | "composing" | "doc-review" | "error"

interface ReviewState {
  reviewId: string
  globalMemories: ProposedMemory[]
  projectMemories: ProposedMemory[] | null
  acceptedCount: number
}

interface DocumentReviewState {
  reviewId: string
  acceptedCount: number
  globalDocument: string | null
  projectDocument: string | null
  globalDiff: string | null
  projectDiff: string | null
}

export type CommandBarState =
  | { phase: "idle" }
  | { phase: "input"; input: string }
  | { phase: "extracting" }
  | ({ phase: "review" } & ReviewState)
  | ({ phase: "composing" } & ReviewState)
  | ({ phase: "doc-review" } & DocumentReviewState)
  | { phase: "error"; error: string; reviewId?: string }

export type MemoryPanelState = Exclude<CommandBarState, { phase: "idle" } | { phase: "input" }>

export interface BoundMemoryActions {
  acceptRemaining: () => Promise<void>
  cancelRemaining: () => Promise<void>
  acceptMemory: (memoryId: string) => Promise<void>
  cancelMemory: (memoryId: string) => Promise<void>
  addMemory: (scope: "global" | "project", memory: string) => void
  setGlobalDocument: (value: string) => void
  setProjectDocument: (value: string) => void
  commitDocuments: () => Promise<void>
}

export interface UseCommandBarReturn {
  state: CommandBarState
  open: () => void
  close: () => void
  setInput: (v: string) => void
  submitCommand: (messages: { role: string; content: string }[], projectSlug?: string | null, customInput?: string) => Promise<void>
  bindProject: (projectSlug?: string | null) => BoundMemoryActions
}

const INITIAL_STATE: CommandBarState = { phase: "idle" }

function isReviewState(state: CommandBarState): state is Extract<CommandBarState, { phase: "review" | "composing" }> {
  return state.phase === "review" || state.phase === "composing"
}

function remainingMemories(state: ReviewState): ProposedMemory[] {
  return [...state.globalMemories, ...(state.projectMemories ?? [])]
}

function removeMemory<T extends ReviewState>(state: T, memoryId: string): T {
  return {
    ...state,
    globalMemories: state.globalMemories.filter((m) => m.id !== memoryId),
    projectMemories: state.projectMemories?.filter((m) => m.id !== memoryId) ?? null,
  }
}

function memoryById(state: ReviewState, memoryId: string): ProposedMemory | undefined {
  return remainingMemories(state).find((m) => m.id === memoryId)
}

export function useCommandBar(): UseCommandBarReturn {
  const [state, setState] = useState<CommandBarState>(INITIAL_STATE)
  const acceptedMemoriesRef = useRef<ProposedMemory[]>([])

  const reset = useCallback(() => {
    acceptedMemoriesRef.current = []
    setState(INITIAL_STATE)
  }, [])

  const open = useCallback(() => {
    acceptedMemoriesRef.current = []
    setState({ phase: "input", input: "" })
  }, [])

  const close = useCallback(() => {
    reset()
  }, [reset])

  const setInput = useCallback((v: string) => {
    setState((prev) => prev.phase === "input" ? { ...prev, input: v } : prev)
  }, [])

  const setGlobalDocument = useCallback((v: string) => {
    setState((prev) => prev.phase === "doc-review" ? { ...prev, globalDocument: v } : prev)
  }, [])

  const setProjectDocument = useCallback((v: string) => {
    setState((prev) => prev.phase === "doc-review" ? { ...prev, projectDocument: v } : prev)
  }, [])

  const composeAccepted = useCallback(async (nextState: ReviewState, projectSlug?: string | null, displayState: ReviewState = nextState) => {
    const acceptedMemories = acceptedMemoriesRef.current
    if (acceptedMemories.length === 0) {
      await cancelMemories(nextState.reviewId, projectSlug)
      reset()
      return
    }

    setState({ ...displayState, acceptedCount: acceptedMemories.length, phase: "composing" })
    try {
      const res: MemoryComposeResponse = await composeMemoryDocs(
        nextState.reviewId,
        acceptedMemories,
        projectSlug,
      )
      setState({
        phase: "doc-review",
        reviewId: nextState.reviewId,
        acceptedCount: acceptedMemories.length,
        globalDocument: res.global_document,
        projectDocument: res.project_document,
        globalDiff: res.global_diff,
        projectDiff: res.project_diff,
      })
    } catch (e) {
      setState({ phase: "error", error: (e as Error)?.message || "Document composition failed", reviewId: nextState.reviewId })
    }
  }, [reset])

  const submitCommand = useCallback(
    async (messages: { role: string; content: string }[], projectSlug?: string | null, customInput?: string) => {
      const rawCmd = customInput !== undefined ? customInput : state.phase === "input" ? state.input : ""
      const cmd = rawCmd.trim()
      if (!cmd) return

      acceptedMemoriesRef.current = []
      setState({ phase: "extracting" })
      if (cmd === "/memorize" || cmd.startsWith("/memorize ")) {
        try {
          const res: MemoryExtractResponse = await extractMemories(messages, projectSlug)
          if (res.global.length === 0 && (!res.project || res.project.length === 0)) {
            setState({ phase: "error", error: "No new memories extracted from this conversation.", reviewId: res.review_id })
            return
          }
          setState({
            phase: "review",
            reviewId: res.review_id,
            globalMemories: res.global,
            projectMemories: res.project,
            acceptedCount: 0,
          })
        } catch (e) {
          setState({ phase: "error", error: (e as Error)?.message || "Extraction failed" })
        }
      } else {
        setState({ phase: "error", error: `Unknown command: ${cmd}` })
      }
    },
    [state],
  )

  const accept = useCallback(
    async (projectSlug?: string | null) => {
      if (!isReviewState(state)) return
      acceptedMemoriesRef.current = [...acceptedMemoriesRef.current, ...remainingMemories(state)]
      const nextState = { ...state, acceptedCount: acceptedMemoriesRef.current.length, globalMemories: [], projectMemories: null }
      await composeAccepted(nextState, projectSlug, state)
    },
    [composeAccepted, state],
  )

  const cancel = useCallback(
    async (projectSlug?: string | null) => {
      if (state.phase === "error") {
        if (state.reviewId) {
          try {
            await cancelMemories(state.reviewId, projectSlug)
          } catch {
            // The pending file may already be gone; close the local UI either way.
          }
        }
        reset()
        return
      }
      if (state.phase === "doc-review") {
        try {
          await cancelMemories(state.reviewId, projectSlug)
        } catch {
          // The pending file may already be gone; close the local UI either way.
        }
        reset()
        return
      }
      if (!isReviewState(state)) return
      if (acceptedMemoriesRef.current.length === 0) {
        try {
          await cancelMemories(state.reviewId, projectSlug)
        } catch {
          // The pending file may already be gone; close the local UI either way.
        }
        reset()
        return
      }
      const nextState = { ...state, globalMemories: [], projectMemories: null }
      await composeAccepted(nextState, projectSlug, state)
    },
    [composeAccepted, reset, state],
  )

  const acceptOne = useCallback(
    async (memoryId: string, projectSlug?: string | null) => {
      if (!isReviewState(state)) return
      const memory = memoryById(state, memoryId)
      if (!memory) return
      acceptedMemoriesRef.current = [...acceptedMemoriesRef.current, memory]
      const nextState = removeMemory({ ...state, acceptedCount: acceptedMemoriesRef.current.length }, memoryId)
      if (remainingMemories(nextState).length === 0) await composeAccepted(nextState, projectSlug, state)
      else setState(nextState)
    },
    [composeAccepted, state],
  )

  const cancelOne = useCallback(
    async (memoryId: string, projectSlug?: string | null) => {
      if (!isReviewState(state)) return
      const nextState = removeMemory(state, memoryId)
      if (remainingMemories(nextState).length === 0) await composeAccepted(nextState, projectSlug, state)
      else setState(nextState)
    },
    [composeAccepted, state],
  )

  const commitDocs = useCallback(
    async (projectSlug?: string | null) => {
      if (state.phase !== "doc-review") return
      try {
        await commitMemoryDocs(state.reviewId, state.globalDocument, state.projectDocument, projectSlug)
        reset()
      } catch (e) {
        setState({ phase: "error", error: (e as Error)?.message || "Commit failed", reviewId: state.reviewId })
      }
    },
    [reset, state],
  )

  const addMemory = useCallback((scope: "global" | "project", memory: string) => {
    const trimmed = memory.trim()
    if (!trimmed) return
    const candidate: ProposedMemory = {
      id: `manual-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      memory: trimmed,
      scope,
      kind: "manual",
      reason: "Added manually",
      categories: ["manual"],
    }
    setState((prev) => {
      if (!isReviewState(prev)) return prev
      return {
        ...prev,
        globalMemories: scope === "global" ? [...prev.globalMemories, candidate] : prev.globalMemories,
        projectMemories: scope === "project" ? [...(prev.projectMemories ?? []), candidate] : prev.projectMemories,
      }
    })
  }, [])

  const bindProject = useCallback((projectSlug?: string | null): BoundMemoryActions => ({
    acceptRemaining: () => accept(projectSlug),
    cancelRemaining: () => cancel(projectSlug),
    acceptMemory: (memoryId: string) => acceptOne(memoryId, projectSlug),
    cancelMemory: (memoryId: string) => cancelOne(memoryId, projectSlug),
    addMemory,
    setGlobalDocument,
    setProjectDocument,
    commitDocuments: () => commitDocs(projectSlug),
  }), [accept, acceptOne, addMemory, cancel, cancelOne, commitDocs, setGlobalDocument, setProjectDocument])

  return {
    state,
    open,
    close,
    setInput,
    submitCommand,
    bindProject,
  }
}
