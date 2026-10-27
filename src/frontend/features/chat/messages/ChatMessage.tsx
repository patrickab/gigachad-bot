"use client"

import dynamic from "next/dynamic"
import { motion } from "framer-motion"
import { memo, useMemo } from "react"
import { Brain, Loader2, RotateCcw, Trash2 } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { CopyAction, ForkAction, TurnAction, TurnRegion } from "@gigachad/chat-ui"
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
  onDelete?: (index: number) => void
  onRegenerate?: (index: number) => void
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

function ChatMessageInner({ role, content, index, onDelete, onRegenerate, onBranch, search_result, research_steps, research_progress, toolRunning, isStreaming, attachments, onAttachmentClick, onCollapse }: ChatMessageProps) {
  const isUser = role === "user"
  const hasResearch = !isUser && !!research_steps?.length
  // Research steps stream into `content` as progress text until the final answer replaces them
  const isResearchRunning = hasResearch && isStreaming
  const canCopy = !!content && !isResearchRunning
  const canRegenerate = !isUser && onRegenerate && !isStreaming && !search_result && !hasResearch
  const canBranch = !isUser && onBranch
  const actions = [
    canCopy && <CopyAction key="copy" text={content} label="Copy message" />,
    canRegenerate && <TurnAction key="regenerate" label="Regenerate response" icon={<RotateCcw size={14} aria-hidden="true" />} onClick={() => onRegenerate(index)} />,
    canBranch && <ForkAction key="branch" label="Branch conversation from this point" onFork={() => onBranch(index)} />,
    onDelete && <TurnAction key="delete" tone="danger" label="Delete message pair" icon={<Trash2 size={14} aria-hidden="true" />} onClick={() => onDelete(index)} />,
  ]

  return (
    <TurnRegion role={isUser ? "question" : "answer"} tail={hasResearch || search_result ? "activity" : "text"} actions={actions} onCollapse={isUser ? onCollapse : undefined}>
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
        <StatusLine>Processing…</StatusLine>
      ) : null}
      {hasResearch && research_steps && (
        <ResearchTrace
          steps={research_steps}
          progress={research_progress}
          isLive={isStreaming}
        />
      )}
    </TurnRegion>
  )
}

export const ChatMessage = memo(ChatMessageInner)
