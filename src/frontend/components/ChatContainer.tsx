"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message, Attachment } from "@/lib/types"
import { ChatMessage } from "./ChatMessage"
import { ChatInput } from "./ChatInput"
import { ChatSidebar } from "./ChatSidebar"
import { getChatSidebarConfig } from "./chatSidebarConfig"
import { ChevronLeft, ChevronRight, User } from "lucide-react"
import { cn } from "@/lib/utils"

const DEFAULT_EXPANDED_TAIL = 2

interface ChatContainerProps {
  chatId: string
  messages: Message[]
  isStreaming: boolean
  onSend: (text: string, attachments: Attachment[]) => void
  onCancel: () => void
  onDeletePair: (index: number) => void
  className?: string
  onOCRRequest?: (image: string) => void
  onRemoveAttachment?: (messageIndex: number, attachmentName: string) => void
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
  onRemoveAttachment,
}: ChatContainerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  const lastMorphicResult = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].morphic_result) return messages[i].morphic_result
    }
    return undefined
  }, [messages])

  const allAttachments = useMemo<{ messageIndex: number; attachment: Attachment }[]>(() => {
    const seen = new Set<string>()
    const entries: { messageIndex: number; attachment: Attachment }[] = []
    for (let i = 0; i < messages.length; i++) {
      const atts = messages[i].attachments
      if (!atts) continue
      for (const att of atts) {
        const key = `${i}-${att.name}`
        if (seen.has(key)) continue
        seen.add(key)
        entries.push({ messageIndex: i, attachment: att })
      }
    }
    return entries
  }, [messages])

  const [contextOpen, setContextOpen] = useState(false)
  const [expandedEntries, setExpandedEntries] = useState<{ messageIndex: number; attachmentName: string }[]>([])
  const [sidebarWidth, setSidebarWidth] = useState(320)
  const [openElements, setOpenElements] = useState<Set<string>>(new Set())

  const handleAttachmentClick = useCallback((messageIndex: number, attachment: Attachment) => {
    setContextOpen(true)
    setExpandedEntries([{ messageIndex, attachmentName: attachment.name }])
  }, [])

  const handleToggleExpand = useCallback((mi: number, name: string) => {
    setExpandedEntries(prev => {
      const idx = prev.findIndex(e => e.messageIndex === mi && e.attachmentName === name)
      if (idx >= 0) {
        return prev.filter((_, i) => i !== idx)
      }
      return [...prev, { messageIndex: mi, attachmentName: name }]
    })
  }, [])

  const handleRemoveAttachment = useCallback((mi: number, name: string) => {
    onRemoveAttachment?.(mi, name)
  }, [onRemoveAttachment])

  useEffect(() => {
    const el = bottomRef.current?.parentElement
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
    if (!nearBottom) return
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const [manualOverrides, setManualOverrides] = useState<Map<number, boolean>>(new Map())

  useEffect(() => {
    if (messages.length === 0) {
      setManualOverrides(new Map())
      setOpenElements(new Set())
    }
  }, [messages])

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
  const tailStartGlobalIndex = pairs.length > DEFAULT_EXPANDED_TAIL
    ? pairs[pairs.length - DEFAULT_EXPANDED_TAIL].globalIndex
    : (pairs.length > 0 ? pairs[0].globalIndex : -1)

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
    return idx >= tailStartGlobalIndex
  }

  function togglePair(idx: number) {
    setManualOverrides(prev => {
      const next = new Map(prev)
      const current = isExpanded(idx)
      next.set(idx, !current)
      return next
    })
  }

  const sidebarElements = useMemo(
    () =>
      getChatSidebarConfig({
        chatId,
        allAttachments,
        expandedEntries,
        onToggleAttachment: handleToggleExpand,
        onRemoveAttachment: handleRemoveAttachment,
        lastMorphicResult,
        isElementOpen: (id) => openElements.has(id),
        onElementOpenChange: (id, open) => {
          setOpenElements((prev) => {
            const next = new Set(prev)
            if (open) next.add(id)
            else next.delete(id)
            return next
          })
        },
      }),
    [chatId, allAttachments, expandedEntries, handleToggleExpand, handleRemoveAttachment, lastMorphicResult, openElements]
  )

  const hasSidebarContent = sidebarElements.length > 0

  useEffect(() => {
    if (!hasSidebarContent) {
      setContextOpen(false)
    }
  }, [hasSidebarContent])

  return (
    <div className={cn("flex h-full relative", className)}>
      <div className="flex-1 min-w-0 flex flex-col relative">
        <div className="flex-1 overflow-y-auto pb-28">
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
                  {!expanded && !isLast && (
                    <button
                      onClick={() => togglePair(globalIndex)}
                      className={cn(
                        "w-full mx-3 my-1.5 flex items-center gap-2.5 px-3 py-2 rounded-lg",
                        "border border-zinc-800/60 bg-zinc-900/30",
                        "text-sm text-zinc-400 hover:bg-zinc-900/70 hover:border-zinc-700 hover:text-zinc-200",
                        "transition-colors group/pair"
                      )}
                      style={{ width: "calc(100% - 1.5rem)" }}
                    >
                      <div className="flex h-7 w-7 items-center justify-center rounded-full bg-zinc-800 shrink-0">
                        <User className="h-3.5 w-3.5 text-zinc-500" />
                      </div>
                      <span className="truncate text-left flex-1">{qLabel}</span>
                      <ChevronRight className="h-3.5 w-3.5 shrink-0 text-zinc-600 group-hover/pair:text-zinc-400" />
                    </button>
                  )}
                  {expanded && (
                    <>
                      <ChatMessage
                        role="user"
                        content={user.content}
                        attachments={user.attachments}
                        index={globalIndex}
                        onAttachmentClick={(att) => handleAttachmentClick(globalIndex, att)}
                        onDelete={onDeletePair}
                        collapsibleUser={!isLast}
                        onCollapse={!isLast ? () => togglePair(globalIndex) : undefined}
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
        <div className="absolute bottom-0 left-0 right-0 z-10 pb-6 pointer-events-none">
          <div className="mx-auto max-w-3xl pointer-events-auto px-4">
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
        {hasSidebarContent && !contextOpen && (
          <button
            onClick={() => setContextOpen(true)}
            title="Open sidebar"
            aria-label="Open sidebar"
            className="absolute right-0 top-3 z-30 flex items-center justify-center h-12 w-4 rounded-l-md border border-r-0 border-zinc-700 bg-zinc-800/80 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 hover:w-5 transition-all shadow-sm"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        )}
        {hasSidebarContent && contextOpen && (
          <button
            onClick={() => setContextOpen(false)}
            title="Collapse sidebar"
            aria-label="Collapse sidebar"
            className="absolute right-0 top-3 z-30 flex items-center justify-center h-12 w-4 rounded-l-md border border-r-0 border-zinc-700 bg-zinc-800/80 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 hover:w-5 transition-all shadow-sm"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        )}
      </div>
      {hasSidebarContent && contextOpen && chatId && (
        <ChatSidebar elements={sidebarElements} width={sidebarWidth} onWidthChange={setSidebarWidth} />
      )}
    </div>
  )
}
