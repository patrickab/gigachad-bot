"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react"
import type { TabConfig } from "@/contexts/SettingsContext"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import type { ChatStreamCompletion, UseChatStreamReturn } from "@/features/chat/useChatStream"
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
import { updateMessageAt } from "@/lib/backend/streaming"
import type { Attachment, ChatRequest, Message, Usage } from "@/lib/domain/types"

type BaseChatRequest = Omit<ChatRequest, "chat_id" | "user_msg" | "img_paths" | "project_slug">

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

/** Send and edit chat turns: attachment intake, tool artifacts, autosave, pair delete and edit, and undoable attachment removal. */
export function useChatTurns({
  chat,
  chatId,
  projectSlug,
  config,
  reasoningEffort,
  prompts,
  autosave,
}: {
  chat: Pick<UseChatStreamReturn, "messages" | "isStreaming" | "totalUsage" | "send" | "removePairs" | "setMessages" | "replaceMessages">
  chatId: string
  projectSlug: string | null
  config: TabConfig
  reasoningEffort: string
  prompts: Record<string, string>
  autosave: (messages: Message[], usage: Usage) => Promise<void>
}) {
  const { messages, isStreaming, totalUsage, send, removePairs, setMessages, replaceMessages } = chat
  const { schedule: scheduleDelete } = useUndoDelete()
  const { enabledTools } = useEnabledTools()
  const baseParams = useMemo(
    () => baseRequest(config, reasoningEffort, prompts, enabledTools),
    [config, reasoningEffort, prompts, enabledTools],
  )

  // Undo restores run up to six seconds later, they must not land in a different chat loaded into this tab meanwhile.
  const chatIdRef = useRef(chatId)
  useEffect(() => { chatIdRef.current = chatId }, [chatId])
  // Every handler reads the messages here: created once per chat, they stay the same while a reply streams, so settled
  // pairs skip its frames
  const messagesRef = useRef(messages)
  const isStreamingRef = useRef(isStreaming)
  const totalUsageRef = useRef(totalUsage)
  useLayoutEffect(() => {
    messagesRef.current = messages
    isStreamingRef.current = isStreaming
    totalUsageRef.current = totalUsage
  }, [messages, isStreaming, totalUsage])

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
    const reads = await readReferences(liveReferencePaths([...messagesRef.current.flatMap((m) => m.attachments ?? []), ...draft]), attachReference)
    return {
      history: replaceMessages((current) => applyReferenceReads(current, reads)),
      draft: draft.map((a) => withReferenceRead(a, reads)),
    }
  }, [replaceMessages])

  /** Re-read `paths` and patch them into whatever the chat holds once the reads finish. */
  const syncReferences = useCallback(async (paths: string[]) => {
    const reads = await readReferences(paths, attachReference)
    setMessages((current) => applyReferenceReads(current, reads))
  }, [setMessages])

  /** Send `text` with `attachments` as a new pair. Resolves to whether the answer completed. */
  const handleSend = useCallback(async (text: string, attachments: Attachment[]): Promise<boolean> => {
    const { history, draft } = await refreshForSend(attachments.map((a) => ({ ...a, active: true })))

    if (draft.length === 0) {
      const completion = await send({
        ...baseParams,
        chat_id: chatId,
        user_msg: text,
        img_paths: collectActiveImagePaths(history),
        project_slug: projectSlug,
      })
      await finishTurn(completion)
      return completion !== null
    }

    replaceMessages((current) => [
      ...current,
      { role: "user" as const, content: text, attachments: draft },
      { role: "assistant" as const, content: "" },
    ])

    const built = await buildAttachedSend({
      text,
      attachments: draft,
      priorMessages: history,
      chatId,
      slug: projectSlug,
      baseParams,
    }, parseFiles).catch(() => null)
    if (!built) {
      updateMessageAt(setMessages, -1, (m) => m.role === "assistant" ? { ...m, content: "Failed to parse attached documents. You can try again or re-upload the file." } : null)
      return false
    }

    updateMessageAt(setMessages, -2, (m) => m.role === "user" ? withAttachments(m, built.enriched) : null)
    const completion = await send(built.request, true)
    await finishTurn(completion)
    return completion !== null
  }, [refreshForSend, finishTurn, send, baseParams, chatId, projectSlug, setMessages, replaceMessages])

  const handleToggleAttachmentActive = useCallback((messageIndex: number, attachmentName: string) => {
    updateMessageAt(setMessages, messageIndex, (msg) => withAttachment(msg, attachmentName, (a) => ({ ...a, active: !a.active })))
  }, [setMessages])

  const handleRemoveAttachment = useCallback((messageIndex: number, attachmentName: string) => {
    const original = messagesRef.current[messageIndex]
    const att = findAttachment(original, attachmentName)
    const stripped = att ? withAttachment(original, attachmentName, () => null) : null
    if (!att || !stripped) return
    const scheduledChat = chatId
    const position = original.attachments?.indexOf(att) ?? 0
    // Removed from the chat (and from what the model sees) at once; the upload goes after the undo window.
    if (!replaceMessages((current) => current.map((m) => m === original ? stripped : m)).includes(stripped)) return
    /** Put `original` back; when the stripped message was replaced meanwhile, re-insert the attachment into the message at `messageIndex` unless it holds one with that name. */
    const restore = () => replaceMessages((current) => {
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
  }, [replaceMessages, chatId, projectSlug, scheduleDelete])

  /** Save the chat after a pair left it. While a reply streams the live pair stays out: its own save follows when it completes. */
  const saveRemoval = useCallback((next: Message[]) => {
    return autosave(isStreamingRef.current ? next.slice(0, -2) : next, totalUsageRef.current)
  }, [autosave])

  /** Delete the pair whose question is at `index`, for good: it leaves the chat and the saved history. */
  const handleDeletePair = useCallback((index: number) => {
    const user = messagesRef.current[index]
    const next = user?.role === "user" ? removePairs(user, "pair") : null
    if (next) void saveRemoval(next)
  }, [removePairs, saveRemoval])

  /**
   * Replace the pair whose question is at `index`, and every later pair, by `text` sent with the question's attachments.
   * Resolves to whether the new answer completed; false without a change while a reply streams.
   */
  const handleEditPair = useCallback(async (index: number, text: string): Promise<boolean> => {
    const user = messagesRef.current[index]
    if (isStreamingRef.current || user?.role !== "user") return false
    const next = removePairs(user, "end")
    if (!next) return false
    await saveRemoval(next)
    return handleSend(text, user.attachments ?? [])
  }, [removePairs, saveRemoval, handleSend])

  /** Save an edit of the attachment's text (see `saveAttachmentEdit`) and rebuild what the model reads. A failed save leaves the chat unchanged and rejects. */
  const handleAttachmentContentChange = useCallback(async (messageIndex: number, attachmentName: string, newContent: string) => {
    const attachment = findAttachment(messagesRef.current[messageIndex], attachmentName)
    if (!attachment) return
    const saved = await saveAttachmentEdit(attachment, newContent)
    updateMessageAt(setMessages, messageIndex, (msg) => withAttachment(msg, attachmentName, () => saved))
  }, [setMessages])

  return {
    handleSend,
    handleToggleAttachmentActive,
    handleRemoveAttachment,
    handleDeletePair,
    handleEditPair,
    handleAttachmentContentChange,
    syncReferences,
  }
}
