"use client"

import { useCallback, useReducer } from "react"
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
  acceptedMemories: ProposedMemory[]
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

export interface UseCommandBarReturn {
  state: CommandBarState
  open: () => void
  close: () => void
  setInput: (v: string) => void
  submitCommand: (messages: { role: string; content: string }[], projectSlug?: string | null, customInput?: string) => Promise<void>
  bindProject: (projectSlug?: string | null) => BoundMemoryActions
}

const INITIAL: CommandBarState = { phase: "idle" }

function isReviewState(state: CommandBarState): state is Extract<CommandBarState, { phase: "review" | "composing" }> {
  return state.phase === "review" || state.phase === "composing"
}

function remaining(state: ReviewState): ProposedMemory[] {
  return [...state.globalMemories, ...(state.projectMemories ?? [])]
}

function findMemory(state: ReviewState, id: string): ProposedMemory | undefined {
  return remaining(state).find((m) => m.id === id)
}

type Action =
  | { type: "OPEN" }
  | { type: "RESET" }
  | { type: "SET_INPUT"; input: string }
  | { type: "START_EXTRACT" }
  | { type: "EXTRACT_OK"; reviewId: string; global: ProposedMemory[]; project: ProposedMemory[] | null }
  | { type: "FAIL"; error: string; reviewId?: string }
  | { type: "DECIDE_ONE"; memoryId: string; accept: boolean }
  | { type: "ACCEPT_ALL" }
  | { type: "ADD_MEMORY"; scope: "global" | "project"; memory: ProposedMemory }
  | { type: "UPDATE_MEMORY"; memoryId: string; updates: Partial<Pick<ProposedMemory, "memory" | "category">> }
  | { type: "BEGIN_PREVIEW"; reviewId: string; acceptedMemories: ProposedMemory[]; rejectedMemories: ProposedMemory[]; loadingScopes: ("global" | "project")[] }
  | { type: "PREVIEW_OK"; scope: "global" | "project"; preview: DocPreview; reviewId: string }
  | { type: "UPDATE_REVISED"; scope: "global" | "project"; memoryId: string; updates: Partial<Pick<PreviewMemory, "text" | "category">> }
  | { type: "REMOVE_REVISED"; scope: "global" | "project"; memoryId: string }

function revisedPreview(state: Extract<CommandBarState, { phase: "doc-review" }>, scope: "global" | "project", fn: (ms: PreviewMemory[]) => PreviewMemory[]): CommandBarState {
  const key = scope === "global" ? "globalPreview" as const : "projectPreview" as const
  const preview = state[key]
  if (!preview) return state
  return { ...state, [key]: { ...preview, revised_memories: fn(preview.revised_memories) } }
}

function reducer(state: CommandBarState, action: Action): CommandBarState {
  switch (action.type) {
    case "OPEN": return { phase: "input", input: "" }
    case "RESET": return INITIAL
    case "SET_INPUT": return state.phase === "input" ? { ...state, input: action.input } : state
    case "START_EXTRACT": return { phase: "extracting" }
    case "EXTRACT_OK": return {
      phase: "review", reviewId: action.reviewId,
      globalMemories: action.global, projectMemories: action.project,
      acceptedMemories: [], rejectedMemories: [],
    }
    case "FAIL": {
      if (state.phase === "doc-review" && action.reviewId && state.reviewId !== action.reviewId) return state
      return { phase: "error", error: action.error, reviewId: action.reviewId }
    }
    case "DECIDE_ONE": {
      if (!isReviewState(state)) return state
      const m = findMemory(state, action.memoryId)
      if (!m) return state
      const id = action.memoryId
      return {
        ...state,
        globalMemories: state.globalMemories.filter(x => x.id !== id),
        projectMemories: state.projectMemories?.filter(x => x.id !== id) ?? null,
        ...(action.accept
          ? { acceptedMemories: [...state.acceptedMemories, m] }
          : { rejectedMemories: [...state.rejectedMemories, m] }),
      }
    }
    case "ACCEPT_ALL": {
      if (!isReviewState(state)) return state
      return { ...state, acceptedMemories: [...state.acceptedMemories, ...remaining(state)], globalMemories: [], projectMemories: null }
    }
    case "ADD_MEMORY": {
      if (!isReviewState(state)) return state
      return {
        ...state,
        globalMemories: action.scope === "global" ? [...state.globalMemories, action.memory] : state.globalMemories,
        projectMemories: action.scope === "project" ? [...(state.projectMemories ?? []), action.memory] : state.projectMemories,
      }
    }
    case "UPDATE_MEMORY": {
      if (!isReviewState(state)) return state
      const up = (list: ProposedMemory[]) => list.map(m => m.id === action.memoryId ? { ...m, ...action.updates } : m)
      return { ...state, globalMemories: up(state.globalMemories), projectMemories: state.projectMemories ? up(state.projectMemories) : null }
    }
    case "BEGIN_PREVIEW": return {
      phase: "doc-review", reviewId: action.reviewId,
      acceptedMemories: action.acceptedMemories, rejectedMemories: action.rejectedMemories,
      globalPreview: null, projectPreview: null, loadingScopes: action.loadingScopes,
    }
    case "PREVIEW_OK": {
      if (state.phase !== "doc-review") return state
      if (action.scope === "global") return { ...state, globalPreview: action.preview, loadingScopes: state.loadingScopes.filter(s => s !== "global") }
      return { ...state, projectPreview: action.preview, loadingScopes: state.loadingScopes.filter(s => s !== "project") }
    }
    case "UPDATE_REVISED": {
      if (state.phase !== "doc-review") return state
      return revisedPreview(state, action.scope, list => list.map(m => m.id === action.memoryId ? { ...m, ...action.updates } : m))
    }
    case "REMOVE_REVISED": {
      if (state.phase !== "doc-review") return state
      return revisedPreview(state, action.scope, list => list.filter(m => m.id !== action.memoryId))
    }
  }
}

export function useCommandBar(): UseCommandBarReturn {
  const [state, dispatch] = useReducer(reducer, INITIAL)

  const open = useCallback(() => dispatch({ type: "OPEN" }), [])
  const close = useCallback(() => dispatch({ type: "RESET" }), [])
  const setInput = useCallback((v: string) => dispatch({ type: "SET_INPUT", input: v }), [])

  const startPreview = useCallback(
    async (accepted: ProposedMemory[], rejected: ProposedMemory[], reviewId: string, projectSlug?: string | null) => {
      if (accepted.length === 0 && rejected.length === 0) {
        await cancelMemories(reviewId, projectSlug).catch(() => {})
        dispatch({ type: "RESET" })
        return
      }
      const byScope = (s: string) => accepted.filter(m => m.scope === s)
      const scopes = [
        ...(byScope("global").length ? ["global" as const] : []),
        ...(byScope("project").length && projectSlug ? ["project" as const] : []),
      ]
      dispatch({ type: "BEGIN_PREVIEW", reviewId, acceptedMemories: accepted, rejectedMemories: rejected, loadingScopes: scopes })
      await Promise.all(scopes.map(async s => {
        try {
          dispatch({ type: "PREVIEW_OK", scope: s, reviewId, preview: await previewMemoryDoc(s, byScope(s), s === "global" ? null : projectSlug ?? null) })
        } catch (e) {
          dispatch({ type: "FAIL", reviewId, error: (e as Error)?.message || "Preview failed" })
        }
      }))
    },
    [],
  )

  const submitCommand = useCallback(
    async (messages: { role: string; content: string }[], projectSlug?: string | null, customInput?: string) => {
      const rawCmd = customInput !== undefined ? customInput : state.phase === "input" ? state.input : ""
      const cmd = rawCmd.trim()
      if (!cmd) return
      dispatch({ type: "START_EXTRACT" })
      if (cmd === "/memorize" || cmd.startsWith("/memorize ")) {
        try {
          const res: MemoryExtractResponse = await extractMemories(messages, projectSlug)
          if (res.global.length === 0 && (!res.project || res.project.length === 0)) {
            dispatch({ type: "FAIL", error: "No new memories extracted from this conversation.", reviewId: res.review_id })
          } else {
            dispatch({ type: "EXTRACT_OK", reviewId: res.review_id, global: res.global, project: res.project })
          }
        } catch (e) {
          dispatch({ type: "FAIL", error: (e as Error)?.message || "Extraction failed" })
        }
      } else {
        dispatch({ type: "FAIL", error: `Unknown command: ${cmd}` })
      }
    },
    [state],
  )

  const accept = useCallback(
    async (projectSlug?: string | null) => {
      if (state.phase === "error") {
        if (state.reviewId) await cancelMemories(state.reviewId, projectSlug).catch(() => {})
        dispatch({ type: "RESET" })
        return
      }
      if (!isReviewState(state)) return
      dispatch({ type: "ACCEPT_ALL" })
      await startPreview([...state.acceptedMemories, ...remaining(state)], state.rejectedMemories, state.reviewId, projectSlug)
    },
    [startPreview, state],
  )

  const cancel = useCallback(
    async (projectSlug?: string | null) => {
      if (state.phase === "idle" || state.phase === "input" || state.phase === "extracting") return
      if ("reviewId" in state && state.reviewId) {
        await cancelMemories(state.reviewId, projectSlug).catch(() => {})
      }
      dispatch({ type: "RESET" })
    },
    [state],
  )

  const decideOne = useCallback(
    async (memoryId: string, doAccept: boolean, projectSlug?: string | null) => {
      if (!isReviewState(state)) return
      const memory = findMemory(state, memoryId)
      if (!memory) return
      dispatch({ type: "DECIDE_ONE", memoryId, accept: doAccept })
      if (remaining(state).filter(m => m.id !== memoryId).length === 0) {
        const acc = doAccept ? [...state.acceptedMemories, memory] : state.acceptedMemories
        const rej = doAccept ? state.rejectedMemories : [...state.rejectedMemories, memory]
        await startPreview(acc, rej, state.reviewId, projectSlug)
      }
    },
    [startPreview, state],
  )

  const commitDocs = useCallback(
    async (projectSlug?: string | null) => {
      if (state.phase !== "doc-review") return
      try {
        const gA = state.acceptedMemories.filter(m => m.scope === "global")
        const gR = state.rejectedMemories.filter(m => m.scope === "global")
        const pA = state.acceptedMemories.filter(m => m.scope === "project")
        const pR = state.rejectedMemories.filter(m => m.scope === "project")
        if (gA.length > 0 || gR.length > 0) {
          await commitMemoryDoc("global", gA, null, state.reviewId, gR, state.globalPreview?.revised_memories)
        }
        if ((pA.length > 0 || pR.length > 0) && projectSlug) {
          await commitMemoryDoc("project", pA, projectSlug, state.reviewId, pR, state.projectPreview?.revised_memories)
        }
        dispatch({ type: "RESET" })
      } catch (e) {
        dispatch({ type: "FAIL", reviewId: state.reviewId, error: (e as Error)?.message || "Commit failed" })
      }
    },
    [state],
  )

  const addMemory = useCallback((scope: "global" | "project", memory: string) => {
    const trimmed = memory.trim()
    if (!trimmed) return
    dispatch({
      type: "ADD_MEMORY", scope,
      memory: { id: `manual-${Date.now()}-${Math.random().toString(36).slice(2)}`, memory: trimmed, scope, category: "manual" },
    })
  }, [])

  const editMemory = useCallback((memoryId: string, text: string) =>
    dispatch({ type: "UPDATE_MEMORY", memoryId, updates: { memory: text } }), [])

  const editMemoryCategory = useCallback((memoryId: string, category: string) =>
    dispatch({ type: "UPDATE_MEMORY", memoryId, updates: { category } }), [])

  const editRevisedMemory = useCallback((scope: "global" | "project", memoryId: string, text: string) =>
    dispatch({ type: "UPDATE_REVISED", scope, memoryId, updates: { text } }), [])

  const editRevisedMemoryCategory = useCallback((scope: "global" | "project", memoryId: string, category: string) =>
    dispatch({ type: "UPDATE_REVISED", scope, memoryId, updates: { category } }), [])

  const removeRevisedMemory = useCallback((scope: "global" | "project", memoryId: string) =>
    dispatch({ type: "REMOVE_REVISED", scope, memoryId }), [])

  const bindProject = useCallback(
    (projectSlug?: string | null): BoundMemoryActions => ({
      acceptRemaining: () => accept(projectSlug),
      cancelRemaining: () => cancel(projectSlug),
      acceptMemory: (id: string) => decideOne(id, true, projectSlug),
      cancelMemory: (id: string) => decideOne(id, false, projectSlug),
      addMemory, editMemory, editMemoryCategory,
      editRevisedMemory, editRevisedMemoryCategory, removeRevisedMemory,
      commitDocuments: () => commitDocs(projectSlug),
    }),
    [accept, addMemory, cancel, commitDocs, decideOne, editMemory, editMemoryCategory, editRevisedMemory, editRevisedMemoryCategory, removeRevisedMemory],
  )

  return { state, open, close, setInput, submitCommand, bindProject }
}
