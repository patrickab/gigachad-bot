"use client"

import dynamic from "next/dynamic"
import { motion } from "framer-motion"
import { memo, useMemo } from "react"
import { Brain, Loader2 } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { TurnAnswer, TurnQuestion } from "@/features/conversation/TurnQuestion"
import { ResearchTrace } from "./ResearchTrace"
import { MessageAttachments } from "./MessageAttachments"
import type { Message, Attachment } from "@/lib/domain/types"

const WebSearchResult = dynamic(() => import("./WebSearchResult").then(m => m.WebSearchResult), { ssr: false })

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
  search_result?: Message["search_result"]
  research_steps?: Message["research_steps"]
  research_progress?: Message["research_progress"]
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
  search_result?: Message["search_result"]
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

/** Spinner plus a short status text for a turn that has nothing to show yet. */
function StatusLine({ children }: { children: string }) {
  return (
    <span role="status" className="inline-flex items-center gap-2 text-ink text-sm">
      <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-faint" aria-hidden="true" />
      <span>{children}</span>
    </span>
  )
}

export function AssistantMessageContent({
  content,
  isStreaming,
  compact,
  search_result,
}: AssistantMessageContentProps) {
  const { thought, cleanContent } = useMemo(() => splitThought(content), [content])

  if (search_result) {
    if (!content.trim() && isStreaming) {
      return <StatusLine>Searching…</StatusLine>
    }
    return <WebSearchResult content={content} search_result={search_result} compact={compact} />
  }

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

function ChatMessageInner({ role, content, index, onDelete, onEdit, laterTurnsOf, onBranch, search_result, research_steps, research_progress, toolRunning, isStreaming, attachments, onAttachmentClick, onCollapse }: ChatMessageProps) {
  const isUser = role === "user"
  const hasResearch = !isUser && !!research_steps?.length
  // Research steps stream into `content` as progress text until the final answer replaces them
  const isResearchRunning = hasResearch && isStreaming
  const body = (
    <>
      {isUser ? (
        <>
          {content && <div><LaTeXMarkdown content={content} /></div>}
          {attachments && attachments.length > 0 && (
            <MessageAttachments attachments={attachments} onClick={onAttachmentClick ?? (() => {})} />
          )}
        </>
      ) : (content || search_result) && !isResearchRunning ? (
        <AssistantMessageContent
          content={content}
          isStreaming={isStreaming}
          search_result={search_result}
        />
      ) : content ? (
        <span role="status" className="inline-flex items-center gap-1 text-ink text-sm">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-ink" aria-hidden="true" />
          {content}
        </span>
      ) : isStreaming && !toolRunning ? (
        <span role="status" aria-label="Waiting for the reply" className="inline-block h-2 w-2 animate-pulse rounded-full bg-ink" />
      ) : null}
      {hasResearch && research_steps && (
        <ResearchTrace
          steps={research_steps}
          progress={research_progress}
          isLive={isStreaming}
        />
      )}
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
      text={isResearchRunning ? "" : content}
      live={isStreaming}
      onBranch={onBranch && (() => onBranch(index))}
      branchLabel="Branch conversation from this point"
      tail={hasResearch || search_result ? "activity" : "text"}
    >
      {body}
    </TurnAnswer>
  )
}

export const ChatMessage = memo(ChatMessageInner)
