"use client"

import { useEffect, useMemo, useRef, type ComponentProps, type RefObject } from "react"
import { File as FileIcon, Folder, MessageSquareText, Puzzle, Sparkles, type LucideIcon } from "lucide-react"
import { withParsedAttachments } from "@/features/chat/attachments"
import { ChatInput, type ChatInputHandle } from "@/features/chat/composer/ChatInput"
import type { CompleteDraft, ComposerCompletion } from "@/features/chat/composer/CompletionMenu"
import { TokenCounter } from "@/features/chat/TokenCounter"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import { ReasoningSelector } from "@/features/settings/ReasoningSelector"
import { parseFiles } from "@/lib/backend/api"
import type { Attachment, OmpView } from "@/lib/domain/types"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { ErrorText } from "@/components/ui/ErrorText"
import { composerImageFile, isComposerOf, mergeRestoredText, ompModelOf, ompPrompt, ompSelectorModels, selectorModelId, shellHint } from "./composer"
import { useOmp } from "./OmpProvider"
import type { Completion } from "@gigachad/chat-ui/protocol"

const COMPLETION_ICONS: Record<Completion["kind"], LucideIcon> = {
  extension: Puzzle, prompt: MessageSquareText, skill: Sparkles, file: FileIcon, directory: Folder,
}
/** An unanswered completion request (frame reloading) resolves empty after this. */
const COMPLETION_TIMEOUT_MS = 3_000
/** Shared by every omp composer: answers go to all of them, the id picks the asker. */
let lastCompletionRequestId = 0

function toComposerCompletion(completion: Completion): ComposerCompletion {
  const isPath = completion.kind === "file" || completion.kind === "directory"
  return { ...completion, icon: COMPLETION_ICONS[completion.kind], mono: isPath }
}

/** Ask the frame for suggestions for `sessionId`'s draft (`undefined`: no composer state yet, no suggestions). */
function useFrameCompletions(sessionId: string | null | undefined): CompleteDraft | undefined {
  const { sendToComposer, onComposerEvent } = useOmp()
  const pendingRef = useRef(new Map<number, (items: Completion[]) => void>())
  useEffect(() => onComposerEvent("composer-completions", (event) => pendingRef.current.get(event.requestId)?.(event.items)), [onComposerEvent])

  return useMemo(() => {
    if (sessionId === undefined) return undefined
    return (text: string, caret: number) => new Promise<ComposerCompletion[]>((resolve) => {
      const requestId = ++lastCompletionRequestId
      const pending = pendingRef.current
      const settle = (items: Completion[]) => {
        pending.delete(requestId)
        window.clearTimeout(timer)
        resolve(items.map(toComposerCompletion))
      }
      const timer = window.setTimeout(() => settle([]), COMPLETION_TIMEOUT_MS)
      pending.set(requestId, settle)
      sendToComposer({ type: "composer-complete", sessionId, requestId, text, caret })
    })
  }, [sessionId, sendToComposer])
}

/** Put what omp hands back for `view` (abort-withdrawn queue, queue Edit) into the composer behind `inputRef`. */
function useComposerRestore(view: OmpView, inputRef: RefObject<ChatInputHandle | null>) {
  const { onComposerEvent } = useOmp()
  useEffect(() => onComposerEvent("composer-restore", (event) => {
    if (!isComposerOf(view, event.sessionId)) return
    inputRef.current?.updateText((draft) => mergeRestoredText(draft, event))
    if (event.images.length > 0) inputRef.current?.addFiles(event.images.map(composerImageFile))
  }), [onComposerEvent, view, inputRef])
}

/**
 * Focus the composer once `view`'s extension `select` closes: the send that opened it disabled the input, which dropped focus.
 * A file the pick opens in Neovim takes focus later, once its terminal attaches.
 */
function useFocusAfterSelect(view: OmpView, inputRef: RefObject<ChatInputHandle | null>) {
  const { pendingSelect } = useOmp()
  const selectOpen = pendingSelect !== null && isComposerOf(view, pendingSelect.sessionId)
  const wasOpenRef = useRef(false)
  useEffect(() => {
    if (wasOpenRef.current && !selectOpen) inputRef.current?.focus()
    wasOpenRef.current = selectOpen
  }, [selectOpen, inputRef])
}

/** The frame's composer state when it shows `view`, null while the frame loads or shows another view. */
function useComposerFor(view: OmpView) {
  const { composer, frameStatus, sendToComposer } = useOmp()
  return { state: frameStatus === "ready" && composer && isComposerOf(view, composer.sessionId) ? composer : null, sendToComposer }
}

/**
 * gigachad's composer for an omp tab. omp-web's own one stays hidden in the frame and submits what this one sends.
 *
 * - Images go as omp images, documents as text the way gigachad chats send them
 * - Text and images omp hands back land in this composer
 * - Disabled until the frame reports the composer of this tab's session
 */
export function OmpComposer({ view, chatId, slug }: { view: OmpView; chatId: string; slug: string | null }) {
  const { state, sendToComposer } = useComposerFor(view)
  const inputRef = useRef<ChatInputHandle>(null)
  useComposerRestore(view, inputRef)
  useFocusAfterSelect(view, inputRef)
  const sessionId = state?.sessionId ?? null
  const complete = useFrameCompletions(state?.sessionId)

  const { run: send, pending: preparing, error } = useAsyncAction(async (text: string, attachments: Attachment[]) => {
    const prompt = await ompPrompt(text, attachments, (documents) => withParsedAttachments(documents, chatId, slug, parseFiles))
    sendToComposer({ type: "composer-send", sessionId, ...prompt })
  }, "Could not attach these files")

  return (
    <div className="px-4 pb-6 pt-2">
      <ChatInput
        ref={inputRef}
        chatId={chatId}
        slug={slug}
        onSend={(text, attachments) => { void send(text, attachments) }}
        disabled={!state || preparing}
        extracting={preparing}
        isStreaming={state?.streaming ?? false}
        onCancel={() => sendToComposer({ type: "composer-abort", sessionId })}
        toolsMenu={false}
        placeholder="Message omp"
        complete={complete}
        hint={shellHint}
      />
      <ErrorText className="mx-auto mt-1 max-w-2xl px-1">{error}</ErrorText>
    </div>
  )
}

type DropdownProps = ComponentProps<typeof ModelDropdown>

/** gigachad's model selector for an omp tab: the OMP provider rows, applied to the session through the frame. */
export function OmpModelSelector({ view, models, ...dropdown }: { view: OmpView } & Omit<DropdownProps, "selectedModel" | "onSelect">) {
  const { state, sendToComposer } = useComposerFor(view)
  if (view.kind === "settings") return null
  const select = (selectorId: string) => {
    const model = ompModelOf(selectorId)
    if (model && state) sendToComposer({ type: "composer-set-model", sessionId: state.sessionId, ...model })
  }
  return (
    <ModelDropdown
      {...dropdown}
      models={models && ompSelectorModels(models)}
      selectedModel={state?.model ? selectorModelId(state.model) : ""}
      onSelect={select}
    />
  )
}

/** gigachad's reasoning selector for an omp tab: omp-web's thinking levels, applied to the session through the frame. */
export function OmpReasoningSelector({ view }: { view: OmpView }) {
  const { state, sendToComposer } = useComposerFor(view)
  if (!state) return null
  return (
    <ReasoningSelector
      levels={state.thinkingLevels}
      reasoningEffort={state.thinkingLevel}
      onReasoningChange={(level) => sendToComposer({ type: "composer-set-thinking", sessionId: state.sessionId, level })}
      placement="below"
    />
  )
}

/** gigachad's token readout for an omp tab: the context tokens omp reports for the session, never a percentage. */
export function OmpTokenCounter({ view }: { view: OmpView }) {
  const { state } = useComposerFor(view)
  return state?.contextTokens ? <TokenCounter tokens={state.contextTokens} /> : null
}
