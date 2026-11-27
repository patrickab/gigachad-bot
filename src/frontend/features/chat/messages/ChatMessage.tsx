"use client"

import { motion } from "framer-motion"
import { memo, useMemo } from "react"
import { Brain } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { TurnAnswer, TurnQuestion } from "@/features/conversation/TurnQuestion"
import { MessageAttachments } from "./MessageAttachments"
import type { Attachment } from "@/lib/domain/types"

const PULSE_DURATION_S = 1.4
const PULSE_SCALE_PEAK = 1.12
const THOUGHT_VERTICAL_SPACING = "my-3"

interface ChatMessageProps {
  role: "user" | "assistant"
  content: string
  index: number
  /** The question's delete, confirmed first. */
  onDelete?: (index: number) => void
  /** The question's edit, confirmed first when later pairs go with it. */
  onEdit?: (index: number, text: string) => void
  /** Counts the pairs after the one at `index`, for the edit's confirmation. */
  laterTurnsOf?: (index: number) => number
  /** The answer's branch. */
  onBranch?: (index: number) => void
  /** A tool call of this message is still running; suppresses the "Processing…" placeholder (ChatContainer renders the tool element). */
  toolRunning?: boolean
  isStreaming?: boolean
  attachments?: Attachment[]
  onAttachmentClick?: (attachment: Attachment) => void
  /** Makes the question's avatar and top strip collapse the pair, alongside a keyboard-accessible chevron. */
  onCollapse?: () => void
}

interface AssistantMessageContentProps {
  content: string
  isStreaming?: boolean
  compact?: boolean
}

/** Split the `<thought>` block from the answer. An unclosed tag means the thought is still streaming. */
function splitThought(content: string): { thought: string | null; cleanContent: string } {
  const startTag = "<thought>\n"
  const endTag = "\n</thought>"
  const startIdx = content.indexOf(startTag)
  if (startIdx === -1) return { thought: null, cleanContent: content }

  const endIdx = content.indexOf(endTag, startIdx + startTag.length)
  const before = content.slice(0, startIdx)

  if (endIdx !== -1) {
    const t = content.slice(startIdx + startTag.length, endIdx).trim() || null
    const clean = (before + content.slice(endIdx + endTag.length)).trim()
    return { thought: t, cleanContent: clean }
  }
  const t = content.slice(startIdx + startTag.length).trim() || null
  return { thought: t, cleanContent: before.trim() }
}

export function AssistantMessageContent({
  content,
  isStreaming,
  compact,
}: AssistantMessageContentProps) {
  const { thought, cleanContent } = useMemo(() => splitThought(content), [content])

  return (
    <div className="text-ink">
      {thought && (
        <details className={`${THOUGHT_VERTICAL_SPACING} pl-4`}>
          <summary className="group/reason cursor-pointer select-none list-none flex items-center gap-1.5 text-[10px] text-ink-subtle hover:text-ink transition-colors marker:text-ink-faint">
            <motion.span
              animate={isStreaming ? { scale: [1, PULSE_SCALE_PEAK, 1] } : { scale: 1 }}
              transition={{ repeat: Infinity, duration: PULSE_DURATION_S, ease: "easeInOut" }}
              className="inline-flex"
            >
              <Brain className="h-3 w-3 text-ink-muted group-hover/reason:text-ink transition-colors" />
            </motion.span>
            <span className="font-medium tracking-wide uppercase group-hover/reason:text-ink transition-colors">Reasoning</span>
          </summary>
          <p className="mt-1 pl-4 border-l-2 border-divider/80 text-xs text-ink-subtle leading-relaxed whitespace-pre-wrap italic">
            {thought}
          </p>
        </details>
      )}
      <LaTeXMarkdown content={cleanContent} streaming={isStreaming} compact={compact} />
    </div>
  )
}

function ChatMessageInner({ role, content, index, onDelete, onEdit, laterTurnsOf, onBranch, toolRunning, isStreaming, attachments, onAttachmentClick, onCollapse }: ChatMessageProps) {
  const isUser = role === "user"
  const body = (
    <>
      {isUser ? (
        <>
          {content && <div data-chat-text={`${index}:user`}><LaTeXMarkdown content={content} /></div>}
          {attachments && attachments.length > 0 && (
            <MessageAttachments attachments={attachments} onClick={onAttachmentClick ?? (() => {})} />
          )}
        </>
      ) : content ? (
        <div data-chat-text={`${index}:assistant`}>
          <AssistantMessageContent
            content={content}
            isStreaming={isStreaming}
          />
        </div>
      ) : isStreaming && !toolRunning ? (
        <span role="status" aria-label="Waiting for the reply" className="inline-block h-2 w-2 animate-pulse rounded-full bg-ink" />
      ) : null}
    </>
  )
  if (isUser) {
    return (
      <TurnQuestion
        text={content}
        live={isStreaming}
        onDelete={onDelete && (() => onDelete(index))}
        onEdit={onEdit && ((text) => onEdit(index, text))}
        laterTurns={laterTurnsOf && (() => laterTurnsOf(index))}
        onCollapse={onCollapse}
      >
        {body}
      </TurnQuestion>
    )
  }
  return (
    <TurnAnswer
      text={content}
      live={isStreaming}
      onBranch={onBranch && (() => onBranch(index))}
      branchLabel="Branch conversation from this point"
      tail="text"
    >
      {body}
    </TurnAnswer>
  )
}

export const ChatMessage = memo(ChatMessageInner)
