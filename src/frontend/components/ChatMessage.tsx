"use client"

import dynamic from "next/dynamic"
import { motion } from "framer-motion"
import { memo, useState } from "react"
import { Bot, Check, Copy, Trash2, User } from "lucide-react"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { ResearchTrace } from "./ResearchTrace"
import { MessageAttachments } from "./MessageAttachments"
import type { Message, Attachment } from "@/lib/types"

const MorphicSearchResult = dynamic(() => import("./MorphicSearchResult").then(m => m.MorphicSearchResult), { ssr: false })

interface ChatMessageProps {
  role: "user" | "assistant"
  content: string
  index: number
  onDelete?: (index: number) => void
  morphic_result?: Message["morphic_result"]
  research_steps?: Message["research_steps"]
  research_progress?: Message["research_progress"]
  research_trace_id?: string
  isStreaming?: boolean
  attachments?: Attachment[]
  messageIndex?: number
  onAttachmentClick?: (attachment: Attachment) => void
}

function ChatMessageInner({ role, content, index, onDelete, morphic_result, research_steps, research_progress, research_trace_id, isStreaming, attachments, onAttachmentClick }: ChatMessageProps) {
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
      transition={{ duration: 0.2 }}
      className={`flex gap-3 px-4 py-3 ${isUser ? "bg-zinc-900/50" : "bg-zinc-950"}`}
    >
      <div className="mt-0.5 shrink-0">
        {isUser ? (
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-zinc-800">
            <User className="h-3.5 w-3.5 text-zinc-400" />
          </div>
        ) : (
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-sky-600/20">
            <Bot className="h-3.5 w-3.5 text-sky-400" />
          </div>
        )}
      </div>
      <div className="min-w-0 flex-1 flex flex-col">
        <div className="mb-0.5 text-xs font-medium text-zinc-500">{isUser ? "You" : "Assistant"}</div>
        {isUser ? (
          <>
            {content && <p className="text-sm whitespace-pre-wrap text-zinc-200">{content}</p>}
            {attachments && attachments.length > 0 && (
              <MessageAttachments attachments={attachments} onClick={onAttachmentClick ?? (() => {})} />
            )}
          </>
        ) : content && !isResearchRunning ? (
          morphic_result ? (
            <MorphicSearchResult content={content} morphic_result={morphic_result} />
          ) : (
            <LaTeXMarkdown content={content} streaming={isStreaming} />
          )
        ) : content ? (
          <span className="inline-flex items-center gap-1 text-zinc-500 text-sm">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-sky-400" />
            {content}
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 text-zinc-500 text-sm">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-sky-400" />
            Thinking…
          </span>
        )}
        {!isUser && (research_steps && research_steps.length > 0) && (
          <ResearchTrace
            steps={research_steps}
            progress={research_progress}
            traceId={research_trace_id}
            isLive={isStreaming}
          />
        )}
        <div className="flex items-center justify-end gap-1 mt-1 opacity-0 transition-opacity group-hover:opacity-100">
          {content && !isResearchRunning && (
            <div className="relative">
              <button
                onClick={handleCopy}
                className="rounded p-1 hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300"
                title="Copy"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-cyan-400" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
              {copied && (
                <motion.span
                  initial={{ opacity: 0, y: 0 }}
                  animate={{ opacity: 1, y: -8 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.4 }}
                  className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] text-cyan-400 pointer-events-none whitespace-nowrap"
                >
                  Copied!
                </motion.span>
              )}
            </div>
          )}
          {onDelete && (
            <button
              onClick={() => onDelete(index)}
              className="rounded p-1 hover:bg-zinc-800 text-zinc-500 hover:text-red-400"
              title="Delete pair"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    </motion.div>
  )
}

export const ChatMessage = memo(ChatMessageInner)
