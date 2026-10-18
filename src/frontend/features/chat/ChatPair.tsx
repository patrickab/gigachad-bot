"use client"

import { ChevronRight, User, Wrench } from "lucide-react"
import type { Attachment, Message } from "@/lib/domain/types"
import { cn } from "@/lib/utils"
import { AssistantMessageContent, ChatMessage } from "@/features/chat/messages/ChatMessage"
import { ElevatedContainer } from "@/components/ui/ElevatedContainer"
import { ToolCallElement } from "@/features/chat/messages/ToolCallElement"
import { TOOL_META } from "@/features/chat/tools"

const MAX_LABEL_LENGTH = 80

const PAIR_SURFACE = {
  backgroundColor: "color-mix(in oklab, var(--surface) 65%, var(--paper))",
  boxShadow: "var(--tw-ring-shadow, 0 0 #0000), var(--shadow-sm)",
}
const RESPONSE_SURFACE = {
  backgroundColor: "color-mix(in oklab, var(--surface-elevated) 45%, var(--surface))",
  boxShadow: "none",
}

/** One-line label for a collapsed question: its first line, else the attachment names. */
function abbreviate(text: string, msg: Message): string {
  const clip = (s: string) => (s.length <= MAX_LABEL_LENGTH ? s : s.slice(0, MAX_LABEL_LENGTH) + "\u2026")
  if (text) return clip(text.split("\n")[0])
  const names = msg.attachments?.map((a) => a.name).join(", ")
  return names ? clip(names) : "(empty)"
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
  /** The collapsed header was clicked. */
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
  return (
    <div data-qa-index={globalIndex} className="group relative">
      {!expanded && (
        <ElevatedContainer
          ref={collapsedRef}
          onClick={onExpand}
          style={PAIR_SURFACE}
          className={cn(
            "mx-3 my-3 rounded-3xl border border-transparent overflow-hidden cursor-pointer",
            focused && "ring-1 ring-ink-muted/50"
          )}
        >
          <div className="flex gap-3 p-3.5 items-center">
            <div className="shrink-0">
              <div className="flex h-7 w-7 items-center justify-center rounded-full bg-surface-elevated">
                <User className="h-3.5 w-3.5 text-ink-muted" />
              </div>
            </div>
            <div className="min-w-0 flex-1 flex flex-col">
              <span className="text-sm text-ink-muted truncate">{abbreviate(user.content, user)}</span>
            </div>
            <ChevronRight className="h-4 w-4 shrink-0 text-ink-faint" />
          </div>
          {(assistant.content || assistant.tool_calls?.length) ? <div className="space-y-1.5 px-3.5 pb-3.5">
            {assistant.tool_calls?.map((call) => {
              const meta = TOOL_META[call.name]
              const Icon = meta?.icon ?? Wrench
              return <ElevatedContainer key={call.id} style={RESPONSE_SURFACE} className="flex items-center gap-2 rounded-[1.125rem] border border-transparent p-3.5 text-xs text-ink-subtle">
                <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>{meta?.cardLabel ?? call.name}</span>
              </ElevatedContainer>
            })}
            {assistant.content && <ElevatedContainer style={RESPONSE_SURFACE} className="rounded-[1.125rem] border border-transparent overflow-hidden">
              <div className="p-3.5">
                <div className="max-h-[12.5vh] overflow-y-auto text-ink-muted">
                  <AssistantMessageContent content={assistant.content} compact />
                </div>
              </div>
            </ElevatedContainer>}
          </div> : null}
        </ElevatedContainer>
      )}
      {expanded && (
        <ElevatedContainer style={PAIR_SURFACE} className={cn("mx-3 my-4 rounded-3xl border border-transparent overflow-hidden", focused && "ring-1 ring-ink-muted/50")}>
          <ChatMessage
            role="user"
            content={user.content}
            attachments={user.attachments}
            index={globalIndex}
            onAttachmentClick={onAttachmentClick}
            onDelete={onDelete}
            collapsibleUser
            onCollapse={onCollapse}
          />
          {/* Same optic as the answer below: the tool call is a peer of the
              response inside the pair, not a decoration on it. */}
          {assistant.tool_calls?.map((call) => (
            <ElevatedContainer key={call.id} style={RESPONSE_SURFACE} className="mx-5 mb-5 rounded-[1.125rem] border border-transparent overflow-hidden">
              <ToolCallElement call={call} />
            </ElevatedContainer>
          ))}
          <ElevatedContainer style={RESPONSE_SURFACE} className="mx-5 mb-5 rounded-[1.125rem] border border-transparent overflow-hidden">
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
          </ElevatedContainer>
        </ElevatedContainer>
      )}
    </div>
  )
}
