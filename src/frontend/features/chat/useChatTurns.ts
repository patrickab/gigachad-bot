"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { TabConfig } from "@/contexts/SettingsContext"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { withoutPairs, type ChatStore, type ChatStreamCompletion } from "@/features/chat/chatStore"
import { deleteAttachment, parseFiles, writeDocument } from "@/lib/backend/api"
import { attachReference } from "@/lib/backend/references"
import {
  applyReferenceReads,
  buildAttachedSend,
  collectActiveImagePaths,
  findAttachment,
  liveReferencePaths,
  readReferences,
  saveAttachmentEdit,
  toolArtifactDocument,
  withAttachment,
  withAttachments,
  withReferenceRead,
} from "@/features/chat/attachments"
import type { Attachment, ChatRequest, Message, Usage } from "@/lib/domain/types"

type BaseChatRequest = Omit<ChatRequest, "chat_id" | "user_msg" | "img_paths" | "project_slug">

/** `messages` with the one at `index` replaced by `change(message)`; the same list when it is missing or `change` gives null. */
function withMessageAt(messages: Message[], index: number, change: (message: Message) => Message | null): Message[] {
  const next = messages[index] && change(messages[index])
  return next ? messages.map((m, i) => i === index ? next : m) : messages
}

/** Build the request fields a tab's config decides. UI-only "none" reasoning maps to null. */
function baseRequest(config: TabConfig, reasoningEffort: string, prompts: Record<string, string>, enabledTools: ChatRequest["tools"]): BaseChatRequest {
  return {
    model: config.selectedModel,
    system_prompt: config.selectedPrompt ? (prompts[config.selectedPrompt] ?? "") : "",
    reasoning_effort: reasoningEffort === "none" ? null : reasoningEffort,
    downscale_images: config.downscaleImages,
    tools: enabledTools,
    tool_options: {
      search_system_instructions: config.searchSystemInstructions ?? "",
      search_domain: config.searchDomain ?? "",
      research_fast_model: config.researchFastModel ?? "",
      research_smart_model: config.researchSmartModel ?? "",
      research_strategic_model: config.researchStrategicModel ?? "",
      research_depth: config.researchDepth,
      research_breadth: config.researchBreadth,
      research_reasoning: config.researchReasoning === "none" ? null : config.researchReasoning,
      research_report_type: config.researchReportType,
    },
  }
}

/** What the reader is told when a pair's delete or an edit's cut could not be saved: the pair stays. */
export const DELETE_UNSAVED = "The chat could not be saved, so the turn was not deleted."
export const EDIT_UNSAVED = "The chat could not be saved, so the turn was not edited."

/**
 * Send and edit chat turns: attachment intake, tool artifacts, autosave, pair delete and edit, and undoable attachment removal.
 *
 * - `save` writes the history file (a no-op without one) and rejects when it fails; `autosave` swallows that
 * - `turnError`: why the last delete or edit left its pair in place, null once a later one (or a send) goes through
 */
export function useChatTurns({
  chat,
  chatId,
  projectSlug,
  config,
  reasoningEffort,
  prompts,
  save,
  autosave,
}: {
  chat: ChatStore
  chatId: string
  projectSlug: string | null
  config: TabConfig
  reasoningEffort: string
  prompts: Record<string, string>
  save: (messages: Message[], usage: Usage) => Promise<void>
  autosave: (messages: Message[], usage: Usage) => Promise<void>
}) {
  const [turnError, setTurnError] = useState<string | null>(null)
  const { schedule: scheduleDelete } = useUndoDelete()
  const { enabledTools } = useEnabledTools()
  const baseParams = useMemo(
    () => baseRequest(config, reasoningEffort, prompts, enabledTools),
    [config, reasoningEffort, prompts, enabledTools],
  )

  // Undo restores run up to six seconds later, they must not land in a different chat loaded into this tab meanwhile.
  const chatIdRef = useRef(chatId)
  useEffect(() => { chatIdRef.current = chatId }, [chatId])

  /** Persist the turn's tool artifacts as documents, then autosave the chat. */
  const finishTurn = useCallback(async (completion: ChatStreamCompletion | null) => {
    if (!completion) return
    const artifacts = (completion.messages.at(-1)?.tool_calls ?? []).flatMap((call) => {
      const artifact = toolArtifactDocument(call)
      return artifact ? [artifact] : []
    })
    const scope = projectSlug ?? ""
    await Promise.allSettled(artifacts.map((artifact) => writeDocument(scope, artifact.name, artifact.content)))
    void autosave(completion.messages, completion.usage)
  }, [projectSlug, autosave])

  /**
   * Re-read the chat's live references and those of `draft`, patched into the latest chat for the next `send`.
   *
   * - Each source is read once, the patch applies to whatever the chat holds once the reads finish
   */
  const refreshForSend = useCallback(async (draft: Attachment[] = []) => {
    const reads = await readReferences(liveReferencePaths([...chat.get().messages.flatMap((m) => m.attachments ?? []), ...draft]), attachReference)
    return {
      history: chat.update((current) => applyReferenceReads(current, reads)),
      draft: draft.map((a) => withReferenceRead(a, reads)),
    }
  }, [chat])

  /** Re-read `paths` and patch them into whatever the chat holds once the reads finish. */
  const syncReferences = useCallback(async (paths: string[]) => {
    const reads = await readReferences(paths, attachReference)
    chat.update((current) => applyReferenceReads(current, reads))
  }, [chat])

  /** Send `text` with `attachments` as a new pair. Resolves to whether the answer completed. */
  const handleSend = useCallback(async (text: string, attachments: Attachment[]): Promise<boolean> => {
    setTurnError(null)
    const { history, draft } = await refreshForSend(attachments.map((a) => ({ ...a, active: true })))

    const answer: Message = { role: "assistant", content: "" }
    if (draft.length === 0) {
      chat.update((current) => [...current, { role: "user", content: text }, answer])
      const completion = await chat.send({
        ...baseParams,
        chat_id: chatId,
        user_msg: text,
        img_paths: collectActiveImagePaths(history),
        project_slug: projectSlug,
      }, answer)
      await finishTurn(completion)
      return completion !== null
    }

    const user: Message = { role: "user", content: text, attachments: draft }
    chat.update((current) => [...current, user, answer])

    const built = await buildAttachedSend({
      text,
      attachments: draft,
      priorMessages: history,
      chatId,
      slug: projectSlug,
      baseParams,
    }, parseFiles).catch(() => null)
    if (!built) {
      chat.update((current) => current.map((m) => m === answer ? { ...m, content: "Failed to parse attached documents. You can try again or re-upload the file." } : m))
      return false
    }

    chat.update((current) => current.map((m) => m === user ? withAttachments(m, built.enriched) : m))
    const completion = await chat.send(built.request, answer)
    await finishTurn(completion)
    return completion !== null
  }, [refreshForSend, finishTurn, chat, baseParams, chatId, projectSlug])

  const handleToggleAttachmentActive = useCallback((messageIndex: number, attachmentName: string) => {
    chat.update((current) => withMessageAt(current, messageIndex, (msg) => withAttachment(msg, attachmentName, (a) => ({ ...a, active: !a.active }))))
  }, [chat])

  const handleRemoveAttachment = useCallback((messageIndex: number, attachmentName: string) => {
    const original = chat.get().messages[messageIndex]
    const att = findAttachment(original, attachmentName)
    const stripped = att ? withAttachment(original, attachmentName, () => null) : null
    if (!att || !stripped) return
    const scheduledChat = chatId
    const position = original.attachments?.indexOf(att) ?? 0
    // Removed from the chat (and from what the model sees) at once; the upload goes after the undo window.
    if (!chat.update((current) => current.map((m) => m === original ? stripped : m)).includes(stripped)) return
    /** Put `original` back; when the stripped message was replaced meanwhile, re-insert the attachment into the message at `messageIndex` unless it holds one with that name. */
    const restore = () => chat.update((current) => {
      if (current.includes(stripped)) return current.map((m) => m === stripped ? original : m)
      const msg = chatIdRef.current === scheduledChat ? current[messageIndex] : undefined
      if (!msg?.attachments || findAttachment(msg, attachmentName)) return current
      const attachments = [...msg.attachments.slice(0, position), att, ...msg.attachments.slice(position)]
      return current.map((m, i) => i === messageIndex ? withAttachments(msg, attachments) : m)
    })
    scheduleDelete(
      `attachment:${chatId}:${messageIndex}:${attachmentName}`,
      attachmentName,
      async () => {
        // File root refs have no upload copy and the source file must never be deleted.
        if (!att.vaultPath) await deleteAttachment(chatId, attachmentName, projectSlug)
      },
      restore,
    )
  }, [chat, chatId, projectSlug, scheduleDelete])

  /**
   * Remove the pair of the question at `index` (`"pair"`), or it and every later one (`"end"`): the history file is written
   * without them first, then they leave the chat. Resolves to the question, or null when nothing changed (no such question,
   * or the save failed: then `turnError` says `unsaved`).
   */
  const removeSaved = useCallback(async (index: number, through: "pair" | "end", unsaved: string): Promise<Message | null> => {
    const { messages, usage } = chat.get()
    const user = messages[index]
    const next = user?.role === "user" ? withoutPairs(messages, user, through) : null
    if (!next) return null
    try {
      await save(next, usage)
    } catch {
      setTurnError(unsaved)
      return null
    }
    setTurnError(null)
    const after = withoutPairs(chat.get().messages, user, through)
    return after && chat.update(() => after) && user
  }, [chat, save])

  /** Delete the pair whose question is at `index`, for good: it leaves the saved history, then the chat. */
  const handleDeletePair = useCallback((index: number) => {
    void removeSaved(index, "pair", DELETE_UNSAVED)
  }, [removeSaved])

  /**
   * Replace the pair whose question is at `index`, and every later pair, by `text` sent with the question's attachments.
   * Resolves to `"sent"` when the new answer completed, `"unsent"` when the pairs went but the send failed, `"kept"` when
   * nothing changed (see `turnError`).
   */
  const handleEditPair = useCallback(async (index: number, text: string): Promise<"sent" | "unsent" | "kept"> => {
    const user = await removeSaved(index, "end", EDIT_UNSAVED)
    if (!user) return "kept"
    return (await handleSend(text, user.attachments ?? []).catch(() => false)) ? "sent" : "unsent"
  }, [removeSaved, handleSend])

  /** Save an edit of the attachment's text (see `saveAttachmentEdit`) and rebuild what the model reads. A failed save leaves the chat unchanged and rejects. */
  const handleAttachmentContentChange = useCallback(async (messageIndex: number, attachmentName: string, newContent: string) => {
    const attachment = findAttachment(chat.get().messages[messageIndex], attachmentName)
    if (!attachment) return
    const saved = await saveAttachmentEdit(attachment, newContent)
    chat.update((current) => withMessageAt(current, messageIndex, (msg) => withAttachment(msg, attachmentName, () => saved)))
  }, [chat])

  return {
    handleSend,
    handleToggleAttachmentActive,
    handleRemoveAttachment,
    handleDeletePair,
    handleEditPair,
    handleAttachmentContentChange,
    syncReferences,
    turnError,
  }
}
