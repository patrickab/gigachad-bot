"use client"

import type { ReactNode } from "react"
import type { Attachment, Message } from "@/lib/domain/types"
import { previewHead, TurnPreview, TurnSurface, usePreviewInteraction } from "@gigachad/chat-ui"
import "@gigachad/chat-ui/turn.css"
import { AssistantMessageContent, ChatMessage } from "@/features/chat/messages/ChatMessage"
import { ToolCallElement } from "@/features/chat/messages/ToolCallElement"
import { toolDisplay } from "@/features/chat/tools"

const MAX_LABEL_LENGTH = 80

/** One-line label for a collapsed question: its first line, else the attachment names. */
export function pairLabel(user: Message): string {
  const clip = (s: string) => (s.length <= MAX_LABEL_LENGTH ? s : s.slice(0, MAX_LABEL_LENGTH) + "\u2026")
  if (user.content) return clip(user.content.split("\n")[0])
  const names = user.attachments?.map((a) => a.name).join(", ")
  return names ? clip(names) : "(empty)"
}

/** The head of a collapsed answer until the reader scrolls or focuses the preview: a collapsed card then holds a fraction of its elements. */
function CollapsedAnswer({ content, search_result }: { content: string; search_result: Message["search_result"] }) {
  const { full, attach } = usePreviewInteraction()
  return <div ref={attach} className="contents"><AssistantMessageContent content={full ? content : previewHead(content)} search_result={search_result} compact /></div>
}

/** A collapsed pair's preview: the tools its answer used and the answer's head, null for an empty answer without tools. The column renders it near the viewport only. */
export function chatPairPreview(assistant: Message): ReactNode {
  return assistant.content || assistant.tool_calls?.length ? <ChatPairPreview assistant={assistant} /> : null
}

export function ChatPairPreview({ assistant }: { assistant: Message }) {
  const tools = (assistant.tool_calls ?? []).map((call) => {
    const { icon: Icon, label } = toolDisplay(call.name)
    return { key: call.id, label, icon: <Icon size={14} aria-hidden="true" /> }
  })
  return (
    <TurnPreview tools={tools}>
      {assistant.content ? <CollapsedAnswer content={assistant.content} search_result={assistant.search_result} /> : null}
    </TurnPreview>
  )
}

interface ChatPairBodyProps {
  user: Message
  assistant: Message
  /** Index of the user message in the full message list. */
  globalIndex: number
  /** Position among the pairs. */
  qaIndex: number
  isStreaming: boolean
  /** Unset for a live pair. */
  onCollapse?: () => void
  onAttachmentClick?: (messageIndex: number, attachment: Attachment) => void
  /** Unset -> read-only pair without a delete action. */
  onDelete?: (index: number) => void
  onRegenerate?: (index: number) => void
  onBranch?: (qaIndex: number) => void
}

/** An open pair's card content: the question, its tool calls and the answer. Props stay equal for a settled pair, so a streamed frame skips it. */
export function ChatPairBody({ user, assistant, globalIndex, qaIndex, isStreaming, onCollapse, onAttachmentClick, onDelete, onRegenerate, onBranch }: ChatPairBodyProps) {
  return (
    <>
      <ChatMessage
        role="user"
        content={user.content}
        attachments={user.attachments}
        index={globalIndex}
        onAttachmentClick={onAttachmentClick && ((attachment) => onAttachmentClick(globalIndex, attachment))}
        onDelete={onDelete}
        onCollapse={onCollapse}
      />
      {assistant.tool_calls?.map((call) => (
        <TurnSurface key={call.id}>
          <ToolCallElement call={call} />
        </TurnSurface>
      ))}
      <TurnSurface>
        <ChatMessage
          role="assistant"
          content={assistant.content}
          search_result={assistant.search_result}
          research_steps={assistant.research_steps}
          research_progress={assistant.research_progress}
          toolRunning={assistant.tool_calls?.some((c) => c.status === "running")}
          isStreaming={isStreaming}
          index={globalIndex}
          onDelete={onDelete}
          onRegenerate={onRegenerate}
          onBranch={onBranch && (() => onBranch(qaIndex))}
        />
      </TurnSurface>
    </>
  )
}
