"use client"

import { useCallback, useRef, useState } from "react"
import {
  cancelMemories,
  commitMemoryDoc,
  extractMemories,
  previewMemoryDoc,
} from "@/lib/api"
import type { MemoryExtractResponse, PreviewMemory, ProposedMemory } from "@/lib/types"

export type CommandBarPhase = "idle" | "input" | "extracting" | "review" | "composing" | "doc-review" | "error"

interface ReviewState {
  reviewId: string
  globalMemories: ProposedMemory[]
  projectMemories: ProposedMemory[] | null
  acceptedCount: number
  rejectedMemories: ProposedMemory[]
}

type DocPreview = { existing_markdown: string; revised_markdown: string; existing_memories: PreviewMemory[]; revised_memories: PreviewMemory[] }

interface DocumentReviewState {
  reviewId: string
  acceptedMemories: ProposedMemory[]
  rejectedMemories: ProposedMemory[]
  globalPreview: DocPreview | null
  projectPreview: DocPreview | null
  loadingScopes: ("global" | "project")[]
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
  editMemory: (memoryId: string, text: string) => void
  editMemoryCategory: (memoryId: string, category: string) => void
  editRevisedMemory: (scope: "global" | "project", memoryId: string, text: string) => void
  editRevisedMemoryCategory: (scope: "global" | "project", memoryId: string, category: string) => void
  removeRevisedMemory: (scope: "global" | "project", memoryId: string) => void
  commitDocuments: () => Promise<void>
}

export interface CommandInputState {
  input: string
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

  const previewAndReview = useCallback(async (nextState: ReviewState, projectSlug?: string | null) => {
    const acceptedMemories = acceptedMemoriesRef.current
    const rejectedMemories = nextState.rejectedMemories

    if (acceptedMemories.length === 0 && rejectedMemories.length === 0) {
      await cancelMemories(nextState.reviewId, projectSlug).catch(() => {})
      reset()
      return
    }

    const globalAccepted = acceptedMemories.filter(m => m.scope === "global")
    const projectAccepted = acceptedMemories.filter(m => m.scope === "project")
    const loadingScopes: ("global" | "project")[] = []
    if (globalAccepted.length > 0) loadingScopes.push("global")
    if (projectAccepted.length > 0 && projectSlug) loadingScopes.push("project")

    setState({
      phase: "doc-review",
      reviewId: nextState.reviewId,
      acceptedMemories,
      rejectedMemories,
      globalPreview: null,
      projectPreview: null,
      loadingScopes,
    })

    const loadPreview = async (scope: "global" | "project", accepted: ProposedMemory[], slug: string | null) => {
      try {
        const preview = await previewMemoryDoc(scope, accepted, slug)
        setState((prev) => {
          if (prev.phase !== "doc-review") return prev
          return {
            ...prev,
            [scope === "global" ? "globalPreview" : "projectPreview"]: preview,
            loadingScopes: prev.loadingScopes.filter((s) => s !== scope),
          }
        })
      } catch (e) {
        setState((prev) =>
          prev.phase === "doc-review" && prev.reviewId === nextState.reviewId
            ? { phase: "error", error: (e as Error)?.message || "Preview failed", reviewId: nextState.reviewId }
            : prev,
        )
      }
    }

    await Promise.all([
      loadingScopes.includes("global") ? loadPreview("global", globalAccepted, null) : Promise.resolve(),
      loadingScopes.includes("project") ? loadPreview("project", projectAccepted, projectSlug ?? null) : Promise.resolve(),
    ])
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
            rejectedMemories: [],
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
      if (state.phase === "error") {
        if (state.reviewId) {
          await cancelMemories(state.reviewId, projectSlug).catch(() => {})
        }
        reset()
        return
      }
      if (!isReviewState(state)) return
      acceptedMemoriesRef.current = [...acceptedMemoriesRef.current, ...remainingMemories(state)]
      const nextState = { ...state, acceptedCount: acceptedMemoriesRef.current.length, globalMemories: [], projectMemories: null }
      await previewAndReview(nextState, projectSlug)
    },
    [previewAndReview, reset, state],
  )

  const cancel = useCallback(
    async (projectSlug?: string | null) => {
      if (state.phase === "error") {
        if (state.reviewId) {
          await cancelMemories(state.reviewId, projectSlug).catch(() => {})
        }
        reset()
        return
      }
      if (state.phase === "doc-review") {
        await cancelMemories(state.reviewId, projectSlug).catch(() => {})
        reset()
        return
      }
      if (!isReviewState(state)) return
      await cancelMemories(state.reviewId, projectSlug).catch(() => {})
      reset()
    },
    [reset, state],
  )

  const acceptOne = useCallback(
    async (memoryId: string, projectSlug?: string | null) => {
      if (!isReviewState(state)) return
      const memory = memoryById(state, memoryId)
      if (!memory) return
      acceptedMemoriesRef.current = [...acceptedMemoriesRef.current, memory]
      const nextState = removeMemory({ ...state, acceptedCount: acceptedMemoriesRef.current.length }, memoryId)
      if (remainingMemories(nextState).length === 0) await previewAndReview(nextState, projectSlug)
      else setState(nextState)
    },
    [previewAndReview, state],
  )

  const cancelOne = useCallback(
    async (memoryId: string, projectSlug?: string | null) => {
      if (!isReviewState(state)) return
      const memory = memoryById(state, memoryId)
      if (!memory) return
      const rejectedMemories = [...state.rejectedMemories, memory]
      const nextState = removeMemory({ ...state, rejectedMemories }, memoryId)
      if (remainingMemories(nextState).length === 0) await previewAndReview(nextState, projectSlug)
      else setState(nextState)
    },
    [previewAndReview, state],
  )

  const commitDocs = useCallback(
    async (projectSlug?: string | null) => {
      if (state.phase !== "doc-review") return
      try {
        const globalAccepted = state.acceptedMemories.filter(m => m.scope === "global")
        const globalRejected = state.rejectedMemories.filter(m => m.scope === "global")
        const projectAccepted = state.acceptedMemories.filter(m => m.scope === "project")
        const projectRejected = state.rejectedMemories.filter(m => m.scope === "project")

        if (globalAccepted.length > 0 || globalRejected.length > 0) {
          await commitMemoryDoc("global", globalAccepted, null, state.reviewId, globalRejected,
            state.globalPreview?.revised_memories)
        }
        if ((projectAccepted.length > 0 || projectRejected.length > 0) && projectSlug) {
          await commitMemoryDoc("project", projectAccepted, projectSlug, state.reviewId, projectRejected,
            state.projectPreview?.revised_memories)
        }
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
      category: "manual",
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

  const editMemory = useCallback((memoryId: string, text: string) => {
    setState((prev) => {
      if (!isReviewState(prev)) return prev
      const updateList = (list: ProposedMemory[]) =>
        list.map((m) => m.id === memoryId ? { ...m, memory: text } : m)
      return {
        ...prev,
        globalMemories: updateList(prev.globalMemories),
        projectMemories: prev.projectMemories ? updateList(prev.projectMemories) : null,
      }
    })
  }, [])

  const editMemoryCategory = useCallback((memoryId: string, category: string) => {
    setState((prev) => {
      if (!isReviewState(prev)) return prev
      const updateList = (list: ProposedMemory[]) =>
        list.map((m) => m.id === memoryId ? { ...m, category } : m)
      return {
        ...prev,
        globalMemories: updateList(prev.globalMemories),
        projectMemories: prev.projectMemories ? updateList(prev.projectMemories) : null,
      }
    })
  }, [])

  const updateRevisedMemories = useCallback(
    (scope: "global" | "project", updater: (list: PreviewMemory[]) => PreviewMemory[]) => {
      setState((prev) => {
        if (prev.phase !== "doc-review") return prev
        const key = scope === "global" ? "globalPreview" : "projectPreview"
        const preview = prev[key]
        if (!preview) return prev
        return { ...prev, [key]: { ...preview, revised_memories: updater(preview.revised_memories) } }
      })
    },
    [],
  )

  const editRevisedMemory = useCallback((scope: "global" | "project", memoryId: string, text: string) => {
    updateRevisedMemories(scope, (list) => list.map((m) => m.id === memoryId ? { ...m, text } : m))
  }, [updateRevisedMemories])

  const editRevisedMemoryCategory = useCallback((scope: "global" | "project", memoryId: string, category: string) => {
    updateRevisedMemories(scope, (list) => list.map((m) => m.id === memoryId ? { ...m, category } : m))
  }, [updateRevisedMemories])

  const removeRevisedMemory = useCallback((scope: "global" | "project", memoryId: string) => {
    updateRevisedMemories(scope, (list) => list.filter((m) => m.id !== memoryId))
  }, [updateRevisedMemories])

  const bindProject = useCallback((projectSlug?: string | null): BoundMemoryActions => ({
    acceptRemaining: () => accept(projectSlug),
    cancelRemaining: () => cancel(projectSlug),
    acceptMemory: (memoryId: string) => acceptOne(memoryId, projectSlug),
    cancelMemory: (memoryId: string) => cancelOne(memoryId, projectSlug),
    addMemory,
    editMemory,
    editMemoryCategory,
    editRevisedMemory,
    editRevisedMemoryCategory,
    removeRevisedMemory,
    commitDocuments: () => commitDocs(projectSlug),
  }), [accept, acceptOne, addMemory, cancel, cancelOne, commitDocs, editMemory, editMemoryCategory, editRevisedMemory, editRevisedMemoryCategory, removeRevisedMemory])

  return {
    state,
    open,
    close,
    setInput,
    submitCommand,
    bindProject,
  }
}
