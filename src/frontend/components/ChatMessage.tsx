"use client"

import { motion } from "framer-motion"
import { memo } from "react"
import { Bot, Copy, Trash2, User } from "lucide-react"
import { MarkdownRenderer } from "./MarkdownRenderer"

interface ChatMessageProps {
  role: "user" | "assistant"
  content: string
  index: number
  onDelete?: (index: number) => void
}

function ChatMessageInner({ role, content, index, onDelete }: ChatMessageProps) {
  const isUser = role === "user"

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
      <div className="min-w-0 flex-1">
        <div className="mb-0.5 text-xs font-medium text-zinc-500">{isUser ? "You" : "Assistant"}</div>
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap text-zinc-200">{content}</p>
        ) : content ? (
          <MarkdownRenderer content={content} />
        ) : (
          <span className="inline-flex items-center gap-1 text-zinc-500 text-sm">
            <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-sky-400" />
            Thinking…
          </span>
        )}
      </div>
      <div className="flex shrink-0 gap-1 opacity-0 transition-opacity group-hover:opacity-100">
        {content && (
          <button
            onClick={() => navigator.clipboard.writeText(content)}
            className="rounded p-1 hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300"
            title="Copy"
          >
            <Copy className="h-3.5 w-3.5" />
          </button>
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
    </motion.div>
  )
}

export const ChatMessage = memo(ChatMessageInner)
