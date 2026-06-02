"use client"

import { useEffect, useRef, useMemo, useState } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message, Attachment } from "@/lib/types"
import { ChatMessage } from "./ChatMessage"
import { ChatInput } from "./ChatInput"
import { SourcesSidebar } from "./SourcesSidebar"
import { AttachmentSidebar } from "./AttachmentSidebar"
import { ChevronRight, ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

interface ChatContainerProps {
  chatId: string
  messages: Message[]
  isStreaming: boolean
  onSend: (text: string, attachments: Attachment[]) => void
  onCancel: () => void
  onDeletePair: (index: number) => void
  className?: string
  onOCRRequest?: (image: string) => void
  activeAttachment?: Attachment | null
  onAttachmentClick?: (attachment: Attachment) => void
  onCloseAttachmentSidebar?: () => void
}

export function ChatContainer({
  chatId,
  messages,
  isStreaming,
  onSend,
  onCancel,
  onDeletePair,
  className,
  onOCRRequest,
  activeAttachment,
  onAttachmentClick,
  onCloseAttachmentSidebar,
}: ChatContainerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  const lastMorphicResult = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].morphic_result) return messages[i].morphic_result
    }
    return undefined
  }, [messages])

  useEffect(() => {
    const el = bottomRef.current?.parentElement
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
    if (!nearBottom) return
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const [manualOverrides, setManualOverrides] = useState<Map<number, boolean>>(new Map())

  useEffect(() => {
    if (messages.length === 0) setManualOverrides(new Map())
  }, [messages])

  const handleAttachmentClick = onAttachmentClick ?? ((_a: Attachment) => {})
  const handleCloseAttachmentSidebar = onCloseAttachmentSidebar ?? (() => {})

  const pairs: { user: Message; assistant: Message; globalIndex: number }[] = []
  for (let i = 0; i < messages.length; i += 2) {
    if (messages[i]?.role === "user") {
      pairs.push({
        user: messages[i],
        assistant: messages[i + 1] ?? { role: "assistant", content: "" },
        globalIndex: i,
      })
    }
  }

  const lastGlobalIndex = pairs.length > 0 ? pairs[pairs.length - 1].globalIndex : -1

  function abbreviate(text: string, msg?: Message): string {
    if (text) {
      const firstLine = text.split("\n")[0]
      if (firstLine.length <= 80) return firstLine
      return firstLine.slice(0, 80) + "\u2026"
    }
    const atts = msg?.attachments
    if (atts && atts.length > 0) {
      const names = atts.map(a => a.name).join(", ")
      return names.length <= 80 ? names : names.slice(0, 80) + "\u2026"
    }
    return "(empty)"
  }

  function isExpanded(idx: number): boolean {
    const override = manualOverrides.get(idx)
    if (override !== undefined) return override
    return idx === lastGlobalIndex
  }

  function togglePair(idx: number) {
    setManualOverrides(prev => {
      const next = new Map(prev)
      const current = isExpanded(idx)
      next.set(idx, !current)
      return next
    })
  }

  return (
    <div className={cn("flex h-full relative", className)}>
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex-1 overflow-y-auto pb-6">
          {pairs.length === 0 && (
            <div className="flex h-full items-center justify-center">
              <p className="text-sm text-zinc-600">Send a message to start.</p>
            </div>
          )}
          <AnimatePresence>
            {pairs.map(({ user, assistant, globalIndex }) => {
              const expanded = isExpanded(globalIndex)
              const isLast = globalIndex === lastGlobalIndex
              const qLabel = abbreviate(user.content, user)
              return (
                <div key={globalIndex} className="group">
                  {!isLast && (
                    <button
                      onClick={() => togglePair(globalIndex)}
                      className="w-full flex items-center gap-2 px-4 py-2 text-sm text-zinc-400 hover:bg-zinc-900/50 transition-colors"
                    >
                      {expanded ? (
                        <ChevronDown className="h-3.5 w-3.5 shrink-0" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5 shrink-0" />
                      )}
                      <span className="truncate text-left">{qLabel}</span>
                    </button>
                  )}
                  {expanded && (
                    <>
                      <ChatMessage
                        role="user"
                        content={user.content}
                        attachments={user.attachments}
                        onAttachmentClick={handleAttachmentClick}
                        index={globalIndex}
                        onDelete={onDeletePair}
                      />
                      <ChatMessage
                        role="assistant"
                        content={assistant.content}
                        morphic_result={assistant.morphic_result}
                        research_steps={assistant.research_steps}
                        research_progress={assistant.research_progress}
                        research_trace_id={assistant.research_trace_id}
                        isStreaming={isStreaming}
                        index={globalIndex}
                        onDelete={onDeletePair}
                      />
                    </>
                  )}
                </div>
              )
            })}
          </AnimatePresence>
          <div ref={bottomRef} />
        </div>
        <div className="shrink-0 px-4 pb-6 z-10">
          <ChatInput
            chatId={chatId}
            onSend={onSend}
            onOCRRequest={onOCRRequest}
            disabled={isStreaming}
            isStreaming={isStreaming}
            onCancel={onCancel}
          />
        </div>
      </div>
      {activeAttachment && chatId && (
        <AttachmentSidebar
          attachment={activeAttachment}
          chatId={chatId}
          onClose={handleCloseAttachmentSidebar}
          isPdf={activeAttachment.mime === "application/pdf"}
        />
      )}
      <SourcesSidebar result={lastMorphicResult} />
    </div>
  )
}