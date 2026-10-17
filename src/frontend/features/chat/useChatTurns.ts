"use client"

import { useCallback, useEffect, useMemo, useRef } from "react"
import type { TabConfig } from "@/contexts/SettingsContext"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import type { ChatStreamCompletion, UseChatStreamReturn } from "@/features/chat/useChatStream"
import { attachRootFile, deleteAttachment, parseFiles, writeDocument } from "@/lib/backend/api"
import {
  buildAttachedSend,
  buildHiddenContent,
  collectActiveImagePaths,
  composeUserMessage,
  toolArtifactDocument,
  withAttachments,
} from "@/features/chat/attachments"
import { updateLastAssistant } from "@/lib/backend/streaming"
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

/** Send, regenerate and edit chat turns: slash commands, attachment intake, tool artifacts, autosave, and undoable pair/attachment removal. */
export function useChatTurns({
  chat,
  chatId,
  activeProject,
  config,
  reasoningEffort,
  prompts,
  autosave,
  refreshDocuments,
  runSlashCommand,
}: {
  chat: Pick<UseChatStreamReturn, "messages" | "isStreaming" | "send" | "regenerateAt" | "deleteMessagePair" | "setMessages">
  chatId: string
  activeProject: string | null
  config: TabConfig
  reasoningEffort: string
  prompts: Record<string, string>
  autosave: (messages: Message[], usage: Usage) => Promise<void>
  refreshDocuments: () => void
  /** Resolves true when `text` was a slash command and has been handled. */
  runSlashCommand: (text: string) => Promise<boolean>
}) {
  const { messages, isStreaming, send, regenerateAt, deleteMessagePair, setMessages } = chat
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
    await Promise.allSettled(artifacts.map((artifact) => writeDocument(activeProject ?? "", artifact.name, artifact.content)))
    refreshDocuments()
    void autosave(completion.messages, completion.usage)
  }, [activeProject, refreshDocuments, autosave])

  const handleSend = useCallback(async (text: string, attachments: Attachment[]) => {
    if (await runSlashCommand(text.trim())) return

    if (attachments.length === 0) {
      await finishTurn(await send({
        ...baseParams,
        chat_id: chatId,
        user_msg: text,
        img_paths: collectActiveImagePaths(messages),
        project_slug: activeProject,
      }))
      return
    }

    const activeAttachments = attachments.map((a) => ({ ...a, active: true }))
    setMessages(prev => [
      ...prev,
      { role: "user" as const, content: text, attachments: activeAttachments },
      { role: "assistant" as const, content: "" },
    ])

    const built = await buildAttachedSend({
      text,
      attachments: activeAttachments,
      priorMessages: messages,
      chatId,
      slug: activeProject,
      baseParams,
    }, parseFiles, attachRootFile).catch(() => null)
    if (!built) {
      updateLastAssistant(setMessages, m => ({ ...m, content: "Failed to parse attached documents. You can try again or re-upload the file." }))
      return
    }

    setMessages(prev => {
      const copy = [...prev]
      const idx = copy.length - 2
      if (idx >= 0 && copy[idx].role === "user") copy[idx] = withAttachments(copy[idx], built.enriched)
      return copy
    })
    await finishTurn(await send(built.request, true))
  }, [runSlashCommand, finishTurn, send, baseParams, chatId, messages, activeProject, setMessages])

  const handleRegenerate = useCallback(async (globalIndex: number) => {
    if (isStreaming) return
    const userMsg = messages[globalIndex]
    if (!userMsg || userMsg.role !== "user") return
    const assistantMsg = messages[globalIndex + 1]
    if (assistantMsg?.search_result || assistantMsg?.research_steps?.length) return

    const attachments = userMsg.attachments ?? []
    const user_msg = attachments.length > 0
      ? composeUserMessage(userMsg.content, userMsg.hiddenContent ?? buildHiddenContent(attachments))
      : userMsg.content
    await finishTurn(await regenerateAt(globalIndex, {
      ...baseParams,
      chat_id: chatId,
      user_msg,
      img_paths: collectActiveImagePaths(messages, { userIndex: globalIndex }),
      project_slug: activeProject,
    }))
  }, [isStreaming, messages, regenerateAt, finishTurn, baseParams, chatId, activeProject])

  const handleToggleAttachmentActive = useCallback((messageIndex: number, attachmentName: string) => {
    setMessages(prev => {
      const msg = prev[messageIndex]
      if (!msg?.attachments) return prev
      const copy = [...prev]
      copy[messageIndex] = withAttachments(msg, msg.attachments.map(a => a.name === attachmentName ? { ...a, active: !a.active } : a))
      return copy
    })
  }, [setMessages])

  const handleRemoveAttachment = useCallback((messageIndex: number, attachmentName: string) => {
    const original = messages[messageIndex]
    const att = original?.attachments?.find(a => a.name === attachmentName)
    if (!original?.attachments || !att) return
    const scheduledChat = chatId
    const patch = ({ attachments, hiddenContent }: Message) => setMessages(prev => {
      if (chatIdRef.current !== scheduledChat || !prev[messageIndex]?.attachments) return prev
      const copy = [...prev]
      copy[messageIndex] = { ...prev[messageIndex], attachments, hiddenContent }
      return copy
    })
    // Removed from the chat (and from what the model sees) at once; the upload goes after the undo window.
    patch(withAttachments(original, original.attachments.filter(a => a.name !== attachmentName)))
    scheduleDelete(
      `attachment:${chatId}:${messageIndex}:${attachmentName}`,
      attachmentName,
      async () => {
        // File root refs have no upload copy and the source file must never be deleted.
        if (!att.vaultPath) await deleteAttachment(chatId, attachmentName, activeProject)
      },
      () => patch(original),
    )
  }, [messages, setMessages, chatId, activeProject, scheduleDelete])

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

  // Edit the text the panel shows (`parsedMd` wins over `content`), and rebuild what the model reads from it.
  const handleAttachmentContentChange = useCallback((messageIndex: number, attachmentName: string, newContent: string) => {
    setMessages(prev => {
      const msg = prev[messageIndex]
      if (!msg?.attachments) return prev
      const copy = [...prev]
      copy[messageIndex] = withAttachments(msg, msg.attachments.map(a =>
        a.name !== attachmentName ? a : a.parsedMd != null ? { ...a, parsedMd: newContent } : { ...a, content: newContent }))
      return copy
    })
  }, [setMessages])

  return {
    handleSend,
    handleRegenerate,
    handleToggleAttachmentActive,
    handleRemoveAttachment,
    handleDeletePair,
    handleAttachmentContentChange,
  }
}
