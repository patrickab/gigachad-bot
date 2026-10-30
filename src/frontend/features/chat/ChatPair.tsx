"use client"

import { useCallback, useState } from "react"
import type { Attachment, Message } from "@/lib/domain/types"
import { TurnCard, TurnHeader, TurnPreview, TurnSurface } from "@gigachad/chat-ui"
import "@gigachad/chat-ui/turn.css"
import { AssistantMessageContent, ChatMessage } from "@/features/chat/messages/ChatMessage"
import { ToolCallElement } from "@/features/chat/messages/ToolCallElement"
import { toolDisplay } from "@/features/chat/tools"

const MAX_LABEL_LENGTH = 80

/** One-line label for a collapsed question: its first line, else the attachment names. */
function abbreviate(text: string, msg: Message): string {
  const clip = (s: string) => (s.length <= MAX_LABEL_LENGTH ? s : s.slice(0, MAX_LABEL_LENGTH) + "\u2026")
  if (text) return clip(text.split("\n")[0])
  const names = msg.attachments?.map((a) => a.name).join(", ")
  return names ? clip(names) : "(empty)"
}

/** Clipping collapsed answers does not avoid parsing their equations. Reserve the
 * preview's 15vh cap offscreen, then use its natural height near the viewport.
 * Short answers can shrink from this estimate, but never stay artificially tall. */
function CollapsedAnswerPreview({ content, search_result }: { content: string; search_result: Message["search_result"] }) {
  const [ready, setReady] = useState(false)

  const attachPreview = useCallback((host: HTMLDivElement | null) => {
    if (!host) return
    // Populate already-visible cards before paint rather than showing a placeholder.
    const rect = host.getBoundingClientRect()
    if (rect.width > 0 && rect.height > 0 && rect.bottom >= -400 && rect.top <= window.innerHeight + 400) {
      setReady(true)
      return
    }
    if (typeof IntersectionObserver === "undefined") {
      setReady(true)
      return
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setReady(true)
        observer.disconnect()
      }
    }, { rootMargin: "400px" })
    observer.observe(host)
    // TurnPreview's focusable region wraps this child; focusing it must also load
    // the preview, even before the intersection callback has been delivered.
    const region = host.parentElement
    const activate = () => setReady(true)
    region?.addEventListener("focusin", activate)
    return () => {
      observer.disconnect()
      region?.removeEventListener("focusin", activate)
    }
  }, [])

  return (
    <div ref={attachPreview} style={ready ? undefined : { height: "15vh" }}>
      {ready
        ? <AssistantMessageContent content={content} search_result={search_result} compact />
        : <span className="sr-only">{content}</span>}
    </div>
  )
}

interface ChatPairProps {
  user: Message
  assistant: Message
  /** Index of the user message in the full message list. */
  globalIndex: number
  expanded: boolean
  /** Keyboard focus is on this pair. */
  focused: boolean
  isStreaming: boolean
  onExpand: () => void
  onCollapse: () => void
  collapsedRef: (el: HTMLDivElement | null) => void
  onAttachmentClick: (attachment: Attachment) => void
  onDelete: (index: number) => void
  onRegenerate?: (index: number) => void
  onBranch?: () => void
}

/** A question and its answer: a collapsed header card, or the expanded messages with their tool calls. */
export function ChatPair({
  user,
  assistant,
  globalIndex,
  expanded,
  focused,
  isStreaming,
  onExpand,
  onCollapse,
  collapsedRef,
  onAttachmentClick,
  onDelete,
  onRegenerate,
  onBranch,
}: ChatPairProps) {
  if (!expanded) {
    const tools = (assistant.tool_calls ?? []).map((call) => {
      const { icon: Icon, label } = toolDisplay(call.name)
      return { key: call.id, label, icon: <Icon size={14} aria-hidden="true" /> }
    })
    return (
      <div data-qa-index={globalIndex}>
        <TurnCard expanded={false} focused={focused} ref={collapsedRef}>
          <TurnHeader label={abbreviate(user.content, user)} onExpand={onExpand} />
          <TurnPreview tools={tools}>
            {assistant.content ? <CollapsedAnswerPreview content={assistant.content} search_result={assistant.search_result} /> : null}
          </TurnPreview>
        </TurnCard>
      </div>
    )
  }

  return (
    <div data-qa-index={globalIndex}>
      <TurnCard expanded focused={focused}>
        <ChatMessage
          role="user"
          content={user.content}
          attachments={user.attachments}
          index={globalIndex}
          onAttachmentClick={onAttachmentClick}
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
            onBranch={onBranch}
          />
        </TurnSurface>
      </TurnCard>
    </div>
  )
}
