"use client"

import { useCallback, useMemo, useReducer } from "react"
import { cancelMemories, commitMemories, extractMemories, getTopics } from "@/lib/backend/api"
import { topicNameFromTitle } from "@/features/memory/memoryUtils"
import { errorMessage } from "@/lib/utils"
import type { Message } from "@/lib/domain/types"
import type { MemoryOperation, MemoryScope, MemoryTopic, ScopeProposal, TopicProposal } from "@/lib/domain/memory"

export interface ReviewOperation extends MemoryOperation { excluded: boolean }
export interface ReviewTopic extends TopicProposal { excluded: boolean }
export interface ReviewScope { operations: ReviewOperation[]; topics: ReviewTopic[] }
type ReviewScopes = Partial<Record<MemoryScope, ReviewScope>>
/** Stored topics per reviewed scope, the targets an operation can move to. */
type StoredTopics = Partial<Record<MemoryScope, MemoryTopic[]>>

/** What one memorization reads: the conversation and the id its extraction watermark is kept under (a chat or an omp session). */
export interface MemorizeSource {
  messages: Pick<Message, "role" | "content">[]
  conversationId: string
}

type MemorizeFlowState =
  | { phase: "idle" }
  | { phase: "extracting" }
  | { phase: "review"; reviewId: string; projectSlug: string | null; scopes: ReviewScopes; storedTopics: StoredTopics; committing: boolean }
  | { phase: "error"; error: string }

export type MemoryPanelState = Exclude<MemorizeFlowState, { phase: "idle" }>

export interface MemoryReviewActions {
  toggleOperation: (scope: MemoryScope, id: string) => void
  editOperation: (scope: MemoryScope, id: string, text: string) => void
  moveOperation: (scope: MemoryScope, id: string, topic: string) => void
  /** Rename a proposed topic, a name already taken merges its operations into that topic. */
  renameTopic: (scope: MemoryScope, name: string, title: string) => void
  editTopicSummary: (scope: MemoryScope, name: string, summary: string) => void
  toggleTopic: (scope: MemoryScope, name: string) => void
  commit: () => Promise<void>
  cancel: () => Promise<void>
}

type Action =
  | { type: "RESET" }
  | { type: "START_EXTRACT" }
  | { type: "EXTRACT_OK"; reviewId: string; projectSlug: string | null; scopes: ReviewScopes; storedTopics: StoredTopics }
  | { type: "FAIL"; error: string }
  | { type: "COMMITTING" }
  | { type: "UPDATE_SCOPE"; scope: MemoryScope; update: (review: ReviewScope) => ReviewScope }

function toReviewScope(proposal: ScopeProposal | null): ReviewScope | undefined {
  if (!proposal || (proposal.operations.length === 0 && proposal.topics.length === 0)) return undefined
  return {
    operations: proposal.operations.map((o) => ({ ...o, excluded: false })),
    topics: proposal.topics.map((t) => ({ ...t, excluded: false })),
  }
}

function reducer(state: MemorizeFlowState, action: Action): MemorizeFlowState {
  switch (action.type) {
    case "RESET": return { phase: "idle" }
    case "START_EXTRACT": return { phase: "extracting" }
    case "EXTRACT_OK": return { phase: "review", reviewId: action.reviewId, projectSlug: action.projectSlug, scopes: action.scopes, storedTopics: action.storedTopics, committing: false }
    case "FAIL": return { phase: "error", error: action.error }
    case "COMMITTING": return state.phase === "review" ? { ...state, committing: true } : state
    case "UPDATE_SCOPE": {
      const review = state.phase === "review" ? state.scopes[action.scope] : undefined
      if (state.phase !== "review" || !review) return state
      return { ...state, scopes: { ...state.scopes, [action.scope]: action.update(review) } }
    }
  }
}

function patchOperation(id: string, patch: Partial<ReviewOperation>) {
  return (review: ReviewScope): ReviewScope => ({
    ...review, operations: review.operations.map((o) => o.id === id ? { ...o, ...patch } : o),
  })
}

function patchTopic(name: string, patch: Partial<ReviewTopic>) {
  return (review: ReviewScope): ReviewScope => ({
    ...review, topics: review.topics.map((t) => t.name === name ? { ...t, ...patch } : t),
  })
}

/** Move every operation of proposed topic `from` to `to`, merging into an existing proposal when one holds that name. */
function renameProposedTopic(from: string, to: string) {
  return (review: ReviewScope): ReviewScope => {
    const merging = review.topics.some((t) => t.name === to)
    return {
      operations: review.operations.map((o) => o.topic === from ? { ...o, topic: to } : o),
      topics: merging ? review.topics.filter((t) => t.name !== from) : review.topics.map((t) => t.name === from ? { ...t, name: to } : t),
    }
  }
}

/** Accepted operations and the topics they need, new topics only while some fact still lands in them. */
export function acceptedChanges(review: ReviewScope) {
  const operations = review.operations.filter((o) => !o.excluded)
  const filled = new Set(operations.filter((o) => o.op !== "delete").map((o) => o.topic))
  const topics = review.topics
    .filter((t) => !t.excluded && (t.previous_summary !== null || filled.has(t.name)))
    .map(({ name, summary }) => ({ name, summary }))
  return { operations, topics }
}

/** Operations across the reviewed scopes, only the accepted ones when `onlyAccepted`. */
export function countOperations(scopes: ReviewScopes, onlyAccepted = false): number {
  return Object.values(scopes).reduce((count, review) => count + (review ? (onlyAccepted ? acceptedChanges(review).operations : review.operations).length : 0), 0)
}

/**
 * Drive the memorize flow: extract proposed operations, review them in one screen, commit.
 *
 * - Every extracted scope commits, even with all operations denied, so denied proposals are not offered again
 * - Cancelling commits nothing and keeps the conversation eligible
 */
export function useMemorizeFlow() {
  const [state, dispatch] = useReducer(reducer, { phase: "idle" })

  const extract = useCallback(async (loadSource: () => Promise<MemorizeSource>, projectSlug: string | null | undefined) => {
    dispatch({ type: "START_EXTRACT" })
    try {
      const { messages, conversationId } = await loadSource()
      const response = await extractMemories(messages.map(({ role, content }) => ({ role, content })), projectSlug, conversationId)
      const scopes: ReviewScopes = { global: toReviewScope(response.global), project: toReviewScope(response.project) }
      if (!scopes.global && !scopes.project) {
        await cancelMemories(response.review_id).catch(() => {})
        dispatch({ type: "FAIL", error: "Nothing new to remember from this conversation." })
        return
      }
      const [globalTopics, projectTopics] = await Promise.all([
        scopes.global ? getTopics("global") : undefined,
        scopes.project ? getTopics("project", projectSlug) : undefined,
      ])
      const storedTopics: StoredTopics = { global: globalTopics?.topics, project: projectTopics?.topics }
      dispatch({ type: "EXTRACT_OK", reviewId: response.review_id, projectSlug: projectSlug ?? null, scopes, storedTopics })
    } catch (e) {
      dispatch({ type: "FAIL", error: errorMessage(e, "Extraction failed") })
    }
  }, [])

  const update = useCallback((scope: MemoryScope, fn: (review: ReviewScope) => ReviewScope) => dispatch({ type: "UPDATE_SCOPE", scope, update: fn }), [])

  const actions = useMemo<MemoryReviewActions>(() => ({
    toggleOperation: (scope, id) => update(scope, (review) => patchOperation(id, {
      excluded: !review.operations.find((o) => o.id === id)?.excluded,
    })(review)),
    editOperation: (scope, id, text) => update(scope, patchOperation(id, { text })),
    moveOperation: (scope, id, topic) => update(scope, patchOperation(id, { topic })),
    renameTopic: (scope, name, title) => {
      const next = topicNameFromTitle(title)
      if (!next || next === name) return
      update(scope, (review) => {
        const renamed = renameProposedTopic(name, next)(review)
        // Renaming onto a stored topic files the facts there, its summary stays as stored.
        const stored = state.phase === "review" ? state.storedTopics[scope] ?? [] : []
        return stored.some((t) => t.name === next) && !review.topics.some((t) => t.name === next)
          ? { ...renamed, topics: renamed.topics.filter((t) => t.name !== next || t.previous_summary !== null) }
          : renamed
      })
    },
    editTopicSummary: (scope, name, summary) => update(scope, patchTopic(name, { summary })),
    toggleTopic: (scope, name) => update(scope, (review) => patchTopic(name, {
      excluded: !review.topics.find((t) => t.name === name)?.excluded,
    })(review)),
    commit: async () => {
      if (state.phase !== "review" || state.committing) return
      dispatch({ type: "COMMITTING" })
      try {
        for (const scope of ["global", "project"] as const) {
          const review = state.scopes[scope]
          if (!review) continue
          const { operations, topics } = acceptedChanges(review)
          await commitMemories(scope, operations, topics, scope === "project" ? state.projectSlug : null, state.reviewId)
        }
        dispatch({ type: "RESET" })
      } catch (e) {
        dispatch({ type: "FAIL", error: errorMessage(e, "Commit failed") })
      }
    },
    cancel: async () => {
      if (state.phase === "review") await cancelMemories(state.reviewId).catch(() => {})
      if (state.phase !== "extracting") dispatch({ type: "RESET" })
    },
  }), [state, update])

  return { state, extract, actions }
}
