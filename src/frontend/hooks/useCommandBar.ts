"use client"

import { useCallback, useState } from "react"
import { extractMemories, acceptMemories, acceptMemory, cancelMemories, cancelMemory } from "@/lib/api"
import type { MemoryExtractResponse, ProposedMemory } from "@/lib/types"

export type CommandBarPhase = "idle" | "input" | "extracting" | "review" | "error"

export interface CommandBarState {
  phase: CommandBarPhase
  input: string
  reviewId: string | null
  globalMemories: ProposedMemory[]
  projectMemories: ProposedMemory[] | null
  error: string | null
}

export interface UseCommandBarReturn {
  state: CommandBarState
  open: () => void
  close: () => void
  setInput: (v: string) => void
  submitCommand: (messages: { role: string; content: string }[], projectSlug?: string | null, customInput?: string) => Promise<void>
  accept: (projectSlug?: string | null) => Promise<void>
  cancel: (projectSlug?: string | null) => Promise<void>
  acceptOne: (memoryId: string) => Promise<void>
  cancelOne: (memoryId: string) => Promise<void>
}

const INITIAL_STATE: CommandBarState = {
  phase: "idle",
  input: "",
  reviewId: null,
  globalMemories: [],
  projectMemories: null,
  error: null,
}

function removeMemory(state: CommandBarState, memoryId: string): CommandBarState {
  const globalMemories = state.globalMemories.filter((m) => m.id !== memoryId)
  const projectMemories = state.projectMemories?.filter((m) => m.id !== memoryId) ?? null
  if (globalMemories.length === 0 && (!projectMemories || projectMemories.length === 0)) return INITIAL_STATE
  return { ...state, globalMemories, projectMemories }
}

export function useCommandBar(): UseCommandBarReturn {
  const [state, setState] = useState<CommandBarState>(INITIAL_STATE)

  const open = useCallback(() => {
    setState({ ...INITIAL_STATE, phase: "input" })
  }, [])

  const close = useCallback(() => {
    setState(INITIAL_STATE)
  }, [])

  const setInput = useCallback((v: string) => {
    setState((prev) => ({ ...prev, input: v }))
  }, [])

  const submitCommand = useCallback(
    async (messages: { role: string; content: string }[], projectSlug?: string | null, customInput?: string) => {
      const rawCmd = customInput !== undefined ? customInput : state.input
      const cmd = rawCmd.trim()
      if (!cmd) return

      setState((prev) => ({ ...prev, input: rawCmd, phase: "extracting", error: null }))
      if (cmd === "/memorize" || cmd.startsWith("/memorize ")) {
        try {
          const res: MemoryExtractResponse = await extractMemories(messages, projectSlug)
          if (res.global.length === 0 && (!res.project || res.project.length === 0)) {
            setState((prev) => ({ ...prev, phase: "error", error: "No new memories extracted from this conversation." }))
            return
          }
          setState((prev) => ({
            ...prev,
            phase: "review",
            reviewId: res.review_id,
            globalMemories: res.global,
            projectMemories: res.project,
          }))
        } catch (e) {
          setState((prev) => ({ ...prev, phase: "error", error: (e as Error)?.message || "Extraction failed" }))
        }
      } else {
        setState((prev) => ({ ...prev, phase: "error", error: `Unknown command: ${cmd}` }))
      }
    },
    [state.input],
  )

  const accept = useCallback(
    async (projectSlug?: string | null) => {
      if (!state.reviewId) return
      try {
        await acceptMemories(state.reviewId, projectSlug)
        setState(INITIAL_STATE)
      } catch (e) {
        setState((prev) => ({ ...prev, phase: "error", error: (e as Error)?.message || "Accept failed" }))
      }
    },
    [state.reviewId],
  )

  const cancel = useCallback(
    async (projectSlug?: string | null) => {
      if (!state.reviewId) return
      try {
        await cancelMemories(state.reviewId, projectSlug)
        setState(INITIAL_STATE)
      } catch (e) {
        setState((prev) => ({ ...prev, phase: "error", error: (e as Error)?.message || "Cancel failed" }))
      }
    },
    [state.reviewId],
  )

  const acceptOne = useCallback(
    async (memoryId: string) => {
      if (!state.reviewId) return
      try {
        await acceptMemory(state.reviewId, memoryId)
        setState((prev) => removeMemory(prev, memoryId))
      } catch (e) {
        setState((prev) => ({ ...prev, phase: "error", error: (e as Error)?.message || "Accept failed" }))
      }
    },
    [state.reviewId],
  )

  const cancelOne = useCallback(
    async (memoryId: string) => {
      if (!state.reviewId) return
      try {
        await cancelMemory(state.reviewId, memoryId)
        setState((prev) => removeMemory(prev, memoryId))
      } catch (e) {
        setState((prev) => ({ ...prev, phase: "error", error: (e as Error)?.message || "Cancel failed" }))
      }
    },
    [state.reviewId],
  )

  return { state, open, close, setInput, submitCommand, accept, cancel, acceptOne, cancelOne }
}
