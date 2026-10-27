"use client"

import { useCallback, useEffect, useMemo, useRef } from "react"
import type { TabConfig } from "@/contexts/SettingsContext"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import type { ChatStreamCompletion, UseChatStreamReturn } from "@/features/chat/useChatStream"
import { attachReference, deleteAttachment, parseFiles, writeDocument } from "@/lib/backend/api"
import {
  applyReferenceReads,
  buildAttachedSend,
  buildHiddenContent,
  collectActiveImagePaths,
  composeUserMessage,
  liveReferencePaths,
  readReferences,
  saveAttachmentEdit,
  toolArtifactDocument,
  withAttachments,
  withReferenceRead,
} from "@/features/chat/attachments"
import { updateLastAssistant } from "@/lib/backend/streaming"
import type { Attachment, ChatRequest, Message, Usage } from "@/lib/domain/types"

type BaseChatRequest = Omit<ChatRequest, "chat_id" | "user_msg" | "img_paths" | "project_slug">

/** Replace the message at `index` (negative counts from the end) with `update(message)`; a missing message or a null result leaves state untouched. */
function updateMessageAt(
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>,
  index: number,
  update: (message: Message) => Message | null,
) {
  setMessages((prev) => {
    const at = index < 0 ? prev.length + index : index
    const next = prev[at] && update(prev[at])
    if (!next) return prev
    const copy = [...prev]
    copy[at] = next
    return copy
  })
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

/** Send, regenerate and edit chat turns: slash commands, attachment intake, tool artifacts, autosave, and undoable pair/attachment removal. */
export function useChatTurns({
  chat,
  chatId,
  projectSlug,
  config,
  reasoningEffort,
  prompts,
  autosave,
  refreshDocuments,
  runSlashCommand,
}: {
  chat: Pick<UseChatStreamReturn, "messages" | "isStreaming" | "send" | "regenerateAt" | "deleteMessagePair" | "setMessages" | "replaceMessages">
  chatId: string
  projectSlug: string | null
  config: TabConfig
  reasoningEffort: string
  prompts: Record<string, string>
  autosave: (messages: Message[], usage: Usage) => Promise<void>
  refreshDocuments: () => void
  /** Resolves true when `text` was a slash command and has been handled. */
  runSlashCommand: (text: string) => Promise<boolean>
}) {
  const { messages, isStreaming, send, regenerateAt, deleteMessagePair, setMessages, replaceMessages } = chat
  const { schedule: scheduleDelete } = useUndoDelete()
  const { enabledTools } = useEnabledTools()
  const baseParams = useMemo(
    () => baseRequest(config, reasoningEffort, prompts, enabledTools),
    [config, reasoningEffort, prompts, enabledTools],
  )

  // Undo restores run up to six seconds later; they must not patch a different chat loaded into this tab meanwhile.
  const chatIdRef = useRef(chatId)
  useEffect(() => { chatIdRef.current = chatId }, [chatId])

  /** Persist the turn's tool artifacts as documents, then autosave the chat. */
  const finishTurn = useCallback(async (completion: ChatStreamCompletion | null) => {
    if (!completion) return
    const artifacts = (completion.messages.at(-1)?.tool_calls ?? []).flatMap((call) => {
      const artifact = toolArtifactDocument(call)
      return artifact ? [artifact] : []
    })
    await Promise.allSettled(artifacts.map((artifact) => writeDocument(projectSlug ?? "", artifact.name, artifact.content)))
    refreshDocuments()
    void autosave(completion.messages, completion.usage)
  }, [projectSlug, refreshDocuments, autosave])

  /**
   * Re-read the chat's live references and those of `draft`, patched into the latest chat for the next `send` or `regenerateAt`.
   *
   * - Each source is read once, the patch applies to whatever the chat holds once the reads finish
   */
  const refreshForSend = useCallback(async (draft: Attachment[] = []) => {
    const reads = await readReferences(liveReferencePaths([...messages.flatMap((m) => m.attachments ?? []), ...draft]), attachReference)
    return {
      history: replaceMessages((current) => applyReferenceReads(current, reads)),
      draft: draft.map((a) => withReferenceRead(a, reads)),
    }
  }, [messages, replaceMessages])

  /** Re-read `paths` and patch them into whatever the chat holds once the reads finish. */
  const syncReferences = useCallback(async (paths: string[]) => {
    const reads = await readReferences(paths, attachReference)
    setMessages((current) => applyReferenceReads(current, reads))
  }, [setMessages])

  const handleSend = useCallback(async (text: string, attachments: Attachment[]) => {
    if (await runSlashCommand(text.trim())) return
    const { history, draft } = await refreshForSend(attachments.map((a) => ({ ...a, active: true })))

    if (draft.length === 0) {
      await finishTurn(await send({
        ...baseParams,
        chat_id: chatId,
        user_msg: text,
        img_paths: collectActiveImagePaths(history),
        project_slug: projectSlug,
      }))
      return
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
      updateLastAssistant(setMessages, m => ({ ...m, content: "Failed to parse attached documents. You can try again or re-upload the file." }))
      return
    }

    updateMessageAt(setMessages, -2, (m) => m.role === "user" ? withAttachments(m, built.enriched) : null)
    await finishTurn(await send(built.request, true))
  }, [runSlashCommand, refreshForSend, finishTurn, send, baseParams, chatId, projectSlug, setMessages, replaceMessages])

  const handleRegenerate = useCallback(async (globalIndex: number) => {
    if (isStreaming) return
    const assistantMsg = messages[globalIndex + 1]
    if (messages[globalIndex]?.role !== "user" || assistantMsg?.search_result || assistantMsg?.research_steps?.length) return
    const { history } = await refreshForSend()
    const userMsg = history[globalIndex]

    const attachments = userMsg.attachments ?? []
    const user_msg = attachments.length > 0
      ? composeUserMessage(userMsg.content, userMsg.hiddenContent ?? buildHiddenContent(attachments))
      : userMsg.content
    await finishTurn(await regenerateAt(globalIndex, {
      ...baseParams,
      chat_id: chatId,
      user_msg,
      img_paths: collectActiveImagePaths(history, { userIndex: globalIndex }),
      project_slug: projectSlug,
    }))
  }, [isStreaming, messages, refreshForSend, regenerateAt, finishTurn, baseParams, chatId, projectSlug])

  const handleToggleAttachmentActive = useCallback((messageIndex: number, attachmentName: string) => {
    updateMessageAt(setMessages, messageIndex, (msg) => msg.attachments
      ? withAttachments(msg, msg.attachments.map(a => a.name === attachmentName ? { ...a, active: !a.active } : a))
      : null)
  }, [setMessages])

  const handleRemoveAttachment = useCallback((messageIndex: number, attachmentName: string) => {
    const original = messages[messageIndex]
    const att = original?.attachments?.find(a => a.name === attachmentName)
    if (!original?.attachments || !att) return
    const scheduledChat = chatId
    const patch = ({ attachments, hiddenContent }: Message) => updateMessageAt(setMessages, messageIndex, (msg) =>
      chatIdRef.current !== scheduledChat || !msg.attachments ? null : { ...msg, attachments, hiddenContent })
    // Removed from the chat (and from what the model sees) at once; the upload goes after the undo window.
    patch(withAttachments(original, original.attachments.filter(a => a.name !== attachmentName)))
    scheduleDelete(
      `attachment:${chatId}:${messageIndex}:${attachmentName}`,
      attachmentName,
      async () => {
        // File root refs have no upload copy and the source file must never be deleted.
        if (!att.vaultPath) await deleteAttachment(chatId, attachmentName, projectSlug)
      },
      () => patch(original),
    )
  }, [messages, setMessages, chatId, projectSlug, scheduleDelete])

  // Nothing to commit: the pair leaves the chat (and what the model sees) at once; Undo puts it back.
  const handleDeletePair = useCallback((index: number) => {
    const user = messages[index]
    const assistant = messages[index + 1]
    if (!user || !assistant) return
    deleteMessagePair(index)
    const scheduledChat = chatId
    scheduleDelete(`pair:${chatId}:${index}`, "message pair", async () => {}, () => setMessages((current) => {
      if (chatIdRef.current !== scheduledChat) return current
      const copy = [...current]
      copy.splice(index, 0, user, assistant)
      return copy
    }))
  }, [messages, deleteMessagePair, setMessages, chatId, scheduleDelete])

  /** Save an edit of the attachment's text (see `saveAttachmentEdit`) and rebuild what the model reads. A failed save leaves the chat unchanged and rejects. */
  const handleAttachmentContentChange = useCallback(async (messageIndex: number, attachmentName: string, newContent: string) => {
    const attachment = messages[messageIndex]?.attachments?.find((a) => a.name === attachmentName)
    if (!attachment) return
    const saved = await saveAttachmentEdit(attachment, newContent)
    updateMessageAt(setMessages, messageIndex, (msg) => msg.attachments
      ? withAttachments(msg, msg.attachments.map(a => a.name === attachmentName ? saved : a))
      : null)
  }, [messages, setMessages])

  return {
    handleSend,
    handleRegenerate,
    handleToggleAttachmentActive,
    handleRemoveAttachment,
    handleDeletePair,
    handleAttachmentContentChange,
    syncReferences,
  }
}
