"use client"

import { useRef, type ComponentProps } from "react"
import { Lock } from "lucide-react"
import { useTab } from "@/contexts/TabContext"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import { ReasoningSelector } from "@/features/settings/ReasoningSelector"
import { displayName } from "@/lib/models"
import type { LearnView } from "@/lib/domain/types"
import { isFrameStateOf, learnSelectorModels, servedModelOf, type LearnModelState } from "./model"
import { useLearn, useLearnDrive, useLearnFrameEvent, useLearnFrameState } from "./LearnProvider"

/** The frame's model pills when it shows `view` in drive mode, null otherwise. A session's model is read-only unless it is no longer served. */
function useModelFor(view: LearnView) {
  const drive = useLearnDrive()
  const model = useLearnFrameState("model")
  const models = useLearnFrameState("models")
  const { sendToLearnFrame } = useLearn()
  const state = drive === "matched" && model && isFrameStateOf(view, model.sessionId) ? model : null
  return { state, served: models?.served ?? null, readOnly: !!state?.locked && !state.unavailable, sendToLearnFrame }
}

/** The session's model and thinking level as a read-only pill: a session keeps the model it started with. */
function SessionModelPill({ state }: { state: LearnModelState }) {
  const name = state.model ? displayName(state.model.modelId) : "No model"
  const level = state.thinkingLevel
  return (
    <span
      role="note"
      aria-label={`Session model: ${name}${level ? `, thinking ${level}` : ""}`}
      className="flex h-8.5 min-w-0 items-center gap-1.5 rounded-full bg-paper px-[11px] text-xs text-ink-muted"
    >
      <Lock className="h-3 w-3 shrink-0" aria-hidden="true" />
      <span className="max-w-[146px] truncate">{name}</span>
      {level && <span className="shrink-0 text-ink-subtle">· {level}</span>}
    </span>
  )
}

type DropdownProps = ComponentProps<typeof ModelDropdown>

/** Open the selector inside `root` unless it is open. */
function openSelectorIn(root: HTMLElement | null): void {
  root?.querySelector<HTMLButtonElement>('button[aria-haspopup="dialog"][aria-expanded="false"]')?.click()
}

/**
 * gigachad's model selector for a learn tab, offering only what LearnHarness's proxy serves.
 *
 * - Home and new session: the pick moves LearnHarness's strong tier, the model new sessions start with
 * - Session: a read-only pill, or the selector again when its model is no longer served
 * - Opens when LearnHarness asks for its model menu (`model-open`: ⌘K in the frame, the send guard)
 * - Manage models edits the full catalog: only the pick list is narrowed to the served models
 */
export function LearnModelSelector({ view, models, ...dropdown }: { view: LearnView } & Omit<DropdownProps, "selectedModel" | "onSelect" | "pickModels">) {
  const { isActive } = useTab()
  const { state, served, readOnly, sendToLearnFrame } = useModelFor(view)
  const rootRef = useRef<HTMLDivElement>(null)
  useLearnFrameEvent("model-open", (event) => {
    if (isActive && isFrameStateOf(view, event.sessionId)) openSelectorIn(rootRef.current)
  })
  if (!state) return null
  if (readOnly) return <SessionModelPill state={state} />

  const select = (selectorId: string) => {
    const model = served && servedModelOf(served, selectorId)
    if (model) sendToLearnFrame({ type: "set-model", sessionId: state.sessionId, ...model })
  }
  return (
    <div ref={rootRef} className="contents">
      <ModelDropdown
        {...dropdown}
        models={served && models}
        pickModels={served ? (catalog) => learnSelectorModels(catalog, served) : undefined}
        selectedModel={state.model?.modelId ?? ""}
        onSelect={select}
      />
    </div>
  )
}

/** gigachad's reasoning selector for a learn tab: the model's thinking levels, applied through the frame. A session's stays in its pill. */
export function LearnReasoningSelector({ view }: { view: LearnView }) {
  const { state, readOnly, sendToLearnFrame } = useModelFor(view)
  if (!state || readOnly || state.thinkingLevel === null) return null
  return (
    <ReasoningSelector
      levels={state.thinkingLevels}
      reasoningEffort={state.thinkingLevel}
      onReasoningChange={(level) => sendToLearnFrame({ type: "set-thinking", sessionId: state.sessionId, level })}
      placement="below"
    />
  )
}
