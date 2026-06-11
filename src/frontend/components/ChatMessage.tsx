"use client"

import dynamic from "next/dynamic"
import { motion } from "framer-motion"
import { memo, useMemo, useState } from "react"
import { Bot, Brain, Check, Copy, GitFork, Loader2, Trash2, User } from "lucide-react"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { ResearchTrace } from "./ResearchTrace"
import { MessageAttachments } from "./MessageAttachments"
import type { Message, Attachment } from "@/lib/types"

const MorphicSearchResult = dynamic(() => import("./MorphicSearchResult").then(m => m.MorphicSearchResult), { ssr: false })

const PULSE_DURATION_S = 1.4
const PULSE_SCALE_PEAK = 1.12
const THOUGHT_VERTICAL_SPACING = "my-3"

interface ChatMessageProps {
  role: "user" | "assistant"
  content: string
  index: number
  onDelete?: (index: number) => void
  onBranch?: (index: number) => void
  morphic_result?: Message["morphic_result"]
  research_steps?: Message["research_steps"]
  research_progress?: Message["research_progress"]
  research_trace_id?: string
  isStreaming?: boolean
  attachments?: Attachment[]
  messageIndex?: number
  onAttachmentClick?: (attachment: Attachment) => void
  collapsibleUser?: boolean
  onCollapse?: () => void
}

export interface AssistantMessageContentProps {
  content: string
  isStreaming?: boolean
  compact?: boolean
  morphic_result?: Message["morphic_result"]
}

export function AssistantMessageContent({
  content,
  isStreaming,
  compact,
  morphic_result,
}: AssistantMessageContentProps) {
  const { thought, cleanContent } = useMemo(() => {
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
  }, [content])

  if (morphic_result) {
    return <MorphicSearchResult content={content} morphic_result={morphic_result} />
  }

  return (
    <div className="text-ink">
      {thought && (
        <details className={`group ${THOUGHT_VERTICAL_SPACING} pl-4`}>
          <summary className="cursor-pointer select-none list-none flex items-center gap-1.5 text-[10px] text-ink-subtle hover:text-ink transition-colors marker:text-ink-faint">
            <motion.span
              animate={isStreaming ? { scale: [1, PULSE_SCALE_PEAK, 1] } : { scale: 1 }}
              transition={{ repeat: Infinity, duration: PULSE_DURATION_S, ease: "easeInOut" }}
              className="inline-flex"
            >
              <Brain className="h-3 w-3 text-ink-muted group-hover:text-ink transition-colors" />
            </motion.span>
            <span className="font-medium tracking-wide uppercase group-hover:text-ink transition-colors">Reasoning</span>
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

function ChatMessageInner({ role, content, index, onDelete, onBranch, morphic_result, research_steps, research_progress, research_trace_id, isStreaming, attachments, onAttachmentClick, collapsibleUser, onCollapse }: ChatMessageProps) {
  const isUser = role === "user"
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  const isResearchRunning = !isUser && isStreaming && research_steps && research_steps.length > 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="flex gap-3 px-6 py-5 text-ink"
    >
      <div
        className={`mt-0.5 shrink-0 ${isUser && collapsibleUser ? "cursor-pointer" : ""}`}
        onClick={isUser && collapsibleUser ? onCollapse : undefined}
      >
        {isUser ? (
          <div className={`flex h-7 w-7 items-center justify-center rounded-xl bg-surface-elevated ${isUser && collapsibleUser ? "hover:bg-surface-elevated transition-colors" : ""}`} title={collapsibleUser ? "Click to collapse" : undefined}>
            <User className="h-3.5 w-3.5 text-ink-muted" />
          </div>
        ) : (
          <div className="flex h-7 w-7 items-center justify-center rounded-xl bg-surface-elevated">
            <Bot className="h-3.5 w-3.5 text-ink" />
          </div>
        )}
      </div>
      <div className="min-w-0 flex-1 flex flex-col">
        <div className="mb-0.5 text-xs font-medium text-ink-subtle">{isUser ? "You" : "Assistant"}</div>
        {isUser ? (
          <>
            {content && <p className="text-sm whitespace-pre-wrap text-ink">{content}</p>}
            {attachments && attachments.length > 0 && (
              <MessageAttachments attachments={attachments} onClick={onAttachmentClick ?? (() => {})} />
            )}
          </>
        ) : content && !isResearchRunning ? (
          <AssistantMessageContent
            content={content}
            isStreaming={isStreaming}
            morphic_result={morphic_result}
          />
        ) : content ? (
          <span role="status" className="inline-flex items-center gap-1 text-ink text-sm">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-ink" aria-hidden="true" />
            {content}
          </span>
        ) : isStreaming ? (
          <span role="status" className="inline-flex items-center gap-2 text-ink text-sm">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-faint" aria-hidden="true" />
            <span>Processing…</span>
          </span>
        ) : null}
        {!isUser && (research_steps && research_steps.length > 0) && (
          <ResearchTrace
            steps={research_steps}
            progress={research_progress}
            traceId={research_trace_id}
            isLive={isStreaming}
          />
        )}
        <div className="flex items-center justify-end gap-1 mt-1 opacity-0 transition-opacity group-hover:opacity-100" role="group" aria-label="Message actions">
          {content && !isResearchRunning && (
            <div className="relative">
              <button
                onClick={(e) => { e.stopPropagation(); handleCopy() }}
              className="rounded p-1 hover:bg-surface-elevated text-ink-subtle hover:text-ink"
              aria-label="Copy message"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-ink" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
              {copied && (
                <motion.span
                  initial={{ opacity: 0, y: 0 }}
                  animate={{ opacity: 1, y: -8 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.4 }}
                  className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] text-ink pointer-events-none whitespace-nowrap"
                >
                  Copied!
                </motion.span>
              )}
            </div>
          )}
          {onDelete && (
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(index) }}
              className="rounded p-1 hover:bg-surface-elevated text-ink-subtle hover:text-danger"
              aria-label="Delete message pair"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
          {!isUser && onBranch && (
            <button
              onClick={(e) => { e.stopPropagation(); onBranch(index) }}
              className="rounded p-1 hover:bg-surface-elevated text-ink-subtle hover:text-ink"
              aria-label="Branch conversation from this point"
            >
              <GitFork className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    </motion.div>
  )
}

export const ChatMessage = memo(ChatMessageInner)
