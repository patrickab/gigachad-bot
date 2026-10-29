"use client"

import { useEffect, useMemo, useRef, type ComponentProps, type RefObject } from "react"
import { File as FileIcon, Folder, MessageSquareText, Puzzle, Sparkles, type LucideIcon } from "lucide-react"
import { withParsedAttachments } from "@/features/chat/attachments"
import { useTab } from "@/contexts/TabContext"
import { ChatInput, type ChatInputHandle } from "@/features/chat/composer/ChatInput"
import type { CompleteDraft, ComposerCompletion, DraftPicker } from "@/features/chat/composer/CompletionMenu"
import { TokenCounter } from "@/features/chat/TokenCounter"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import { ReasoningSelector } from "@/features/settings/ReasoningSelector"
import { parseFiles } from "@/lib/backend/api"
import type { Attachment, OmpView } from "@/lib/domain/types"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { ErrorText } from "@/components/ui/ErrorText"
import {
  composerImageFile, isComposerOf, mergeRestoredText, offersSkills, ompModelOf, ompPrompt, ompSelectorModels, optionParts, rankByName, rankOptions, selectorModelId,
  shellHint, SKILL_PREFIX, skillQueryOf, slashQueryOf, type ComposerCommands,
} from "./composer"
import { useOmp } from "./OmpProvider"
import type { Completion, SlashCommand } from "@gigachad/chat-ui/protocol"

const FILE_ICONS: Record<Completion["kind"], LucideIcon> = { file: FileIcon, directory: Folder }
const COMMAND_ICONS: Record<SlashCommand["kind"], LucideIcon> = { extension: Puzzle, prompt: MessageSquareText }
/** An unanswered completion request (frame reloading) resolves empty after this. */
const COMPLETION_TIMEOUT_MS = 3_000
/** Shared by every omp composer: answers go to all of them, the id picks the asker. */
let lastCompletionRequestId = 0

function toComposerCompletion(completion: Completion): ComposerCompletion {
  return { ...completion, icon: FILE_ICONS[completion.kind], mono: true }
}

/**
 * The `/` menu for `query`, the whole draft replaced by the pick.
 *
 * - `/skill:` first, then the other commands. Picking it keeps the menu open on the skills, filtered by what follows `skill:`
 */
function slashCompletions(text: string, query: string, commands: ComposerCommands | null): ComposerCompletion[] {
  const replaceDraft = { from: 0, to: text.length }
  const skills = commands?.skills ?? []
  const skillQuery = skillQueryOf(query)
  if (skillQuery !== null) {
    return rankByName(skills, skillQuery).map((skill) => {
      const insert = `/${SKILL_PREFIX}${skill.name} `
      return { label: `/${SKILL_PREFIX}${skill.name}`, detail: skill.description, icon: Sparkles, ...replaceDraft, insert, caretOffset: insert.length }
    })
  }
  const skillEntry: ComposerCompletion[] = skills.length > 0 && offersSkills(query)
    ? [{ label: `/${SKILL_PREFIX}`, detail: `${skills.length} skills`, icon: Sparkles, ...replaceDraft, insert: `/${SKILL_PREFIX}`, caretOffset: SKILL_PREFIX.length + 1 }]
    : []
  return [...skillEntry, ...rankByName(commands?.commands ?? [], query).map((command) => {
    const insert = `/${command.name} `
    return { label: `/${command.name}`, detail: command.description, icon: COMMAND_ICONS[command.kind], ...replaceDraft, insert, caretOffset: insert.length }
  })]
}

/** Ask the frame for `@` suggestions for `sessionId`'s draft (`undefined`: no composer state yet, no suggestions). */
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

/**
 * Suggestions for the draft: the `/` menu from the reported commands, filtered here so typing never waits on the frame, else the frame's `@` files.
 *
 * - The first `/` of a session asks the frame to load its commands (`composer-load-commands`), the open menu follows once they arrive
 */
function useComposerCompletions(sessionId: string | null | undefined, commands: ComposerCommands | null): CompleteDraft | undefined {
  const { sendToComposer } = useOmp()
  const frameComplete = useFrameCompletions(sessionId)
  const loadRequestedRef = useRef(new Set<string>())
  return useMemo(() => {
    if (!frameComplete) return undefined
    return async (text: string, caret: number) => {
      const query = slashQueryOf(text)
      if (query === null) return frameComplete(text, caret)
      if (sessionId && !loadRequestedRef.current.has(sessionId)) {
        loadRequestedRef.current.add(sessionId)
        sendToComposer({ type: "composer-load-commands", sessionId })
      }
      return slashCompletions(text, query, commands)
    }
  }, [frameComplete, sessionId, commands, sendToComposer])
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

/** `view`'s pending extension `select` (e.g. `/files`) as the active tab's draft picker, answered through the frame. A path option shows its folder beside it. */
function useSelectPicker(view: OmpView): DraftPicker | undefined {
  const { isActive } = useTab()
  const { pendingSelect, answerSelect } = useOmp()
  const request = isActive && pendingSelect && isComposerOf(view, pendingSelect.sessionId) ? pendingSelect.request : null
  return useMemo(() => {
    if (!request) return undefined
    return {
      label: request.title,
      items: (query: string) => rankOptions(request.options, query).map((option) => {
        const { name, lead } = optionParts(option)
        return { key: option, label: name, detail: lead || undefined, icon: lead ? FileIcon : undefined }
      }),
      onPick: answerSelect,
      onCancel: () => answerSelect(null),
    }
  }, [request, answerSelect])
}

/** The frame's composer state and commands when it shows `view`, null while the frame loads or shows another view. */
function useComposerFor(view: OmpView) {
  const { composer, composerCommands, frameStatus, sendToComposer } = useOmp()
  const shown = frameStatus === "ready"
  return {
    state: shown && composer && isComposerOf(view, composer.sessionId) ? composer : null,
    commands: shown && composerCommands && isComposerOf(view, composerCommands.sessionId) ? composerCommands : null,
    sendToComposer,
  }
}

/**
 * gigachad's composer for an omp tab. omp-web's own one stays hidden in the frame and submits what this one sends.
 *
 * - Images go as omp images, documents as text the way gigachad chats send them
 * - Text and images omp hands back land in this composer
 * - `/skill:` drills into the skills in the same menu, a pick becomes the draft `/skill:<name> `
 * - An extension's `select` (e.g. `/files`) takes over the same menu, the draft filtering its options
 * - Disabled until the frame reports the composer of this tab's session
 */
export function OmpComposer({ view, chatId, slug }: { view: OmpView; chatId: string; slug: string | null }) {
  const { state, commands, sendToComposer } = useComposerFor(view)
  const inputRef = useRef<ChatInputHandle>(null)
  useComposerRestore(view, inputRef)
  const picker = useSelectPicker(view)
  const sessionId = state?.sessionId ?? null
  const complete = useComposerCompletions(state?.sessionId, commands)

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
        picker={picker}
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
