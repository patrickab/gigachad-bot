"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message, Attachment } from "@/lib/types"
import { ChatMessage } from "./ChatMessage"
import { ChatInput } from "./ChatInput"
import { ChatSidebar, MAX_SIDEBAR_WIDTH } from "./ChatSidebar"
import { getChatSidebarConfig } from "./chatSidebarConfig"
import { ChevronLeft, ChevronRight, User } from "lucide-react"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
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

  const [hoveredPair, setHoveredPair] = useState<number | null>(null)
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const leaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleMouseEnter = (globalIndex: number) => {
    if (leaveTimeoutRef.current) {
      clearTimeout(leaveTimeoutRef.current)
      leaveTimeoutRef.current = null
    }
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
    hoverTimeoutRef.current = setTimeout(() => {
      setHoveredPair(globalIndex)
    }, 500)
  }

  const handleMouseLeave = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current)
      hoverTimeoutRef.current = null
    }
    if (leaveTimeoutRef.current) clearTimeout(leaveTimeoutRef.current)
    leaveTimeoutRef.current = setTimeout(() => {
      setHoveredPair(null)
      leaveTimeoutRef.current = null
    }, 100)
  }

  // Clean up timeouts on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
      if (leaveTimeoutRef.current) clearTimeout(leaveTimeoutRef.current)
    }
  }, [])

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
  const containerRef = useRef<HTMLDivElement>(null)
  const inputAreaRef = useRef<HTMLDivElement>(null)
  const [measuredWidth, setMeasuredWidth] = useState(0)
  const [inputAreaHeight, setInputAreaHeight] = useState(0)
  const chatMaxWidth = measuredWidth > 0 ? Math.max(0, measuredWidth - MAX_SIDEBAR_WIDTH) : undefined

  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    setMeasuredWidth(el.clientWidth)
  }, [])

  useEffect(() => {
    const el = inputAreaRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setInputAreaHeight(entry.contentRect.height)
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!hasSidebarContent) {
      setContextOpen(false)
    }
  }, [hasSidebarContent])

  return (
    <div ref={containerRef} className={cn("flex h-full relative", className)}>
      <div className="flex-1 min-w-0 flex flex-col relative">
        <div className="flex-1 overflow-y-auto" style={{ paddingBottom: Math.max(192, inputAreaHeight + 24) }}>
          <div className="mx-auto" style={{ maxWidth: chatMaxWidth || undefined }}>
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
              const showPreview = hoveredPair === globalIndex
              return (
                <div key={globalIndex} className="group relative">
                  {!expanded && !isLast && (
                    <div
                      onMouseEnter={() => handleMouseEnter(globalIndex)}
                      onMouseLeave={handleMouseLeave}
                      onClick={() => togglePair(globalIndex)}
                      className="mx-3 my-4 rounded-xl border border-zinc-800/40 bg-assistant-message shadow-sm overflow-hidden cursor-pointer transition-all duration-300"
                    >
                      <div className="flex gap-3 px-5 py-4 items-start">
                        <div className="mt-0.5 shrink-0">
                          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-zinc-800">
                            <User className="h-3.5 w-3.5 text-zinc-400" />
                          </div>
                        </div>
                        <div className="min-w-0 flex-1 flex flex-col">
                          <div className="mb-0.5 text-xs font-medium text-zinc-500">You</div>
                          {showPreview ? (
                            <div className="max-h-[15vh] overflow-y-auto">
                              <LaTeXMarkdown content={user.content} compact />
                            </div>
                          ) : (
                            <span className="text-sm text-zinc-200 truncate">{qLabel}</span>
                          )}
                        </div>
                        <ChevronRight className="h-4 w-4 shrink-0 text-zinc-600 mt-1" />
                      </div>
                      <div
                        className={cn(
                          "grid transition-[grid-template-rows] duration-300 ease-out",
                          showPreview ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
                        )}
                      >
                        <div className="overflow-hidden">
                          <div className="px-5 pb-4">
                            {assistant.content && (
                              <div className="rounded-lg bg-zinc-900 border border-zinc-800/30 shadow-[0_-8px_24px_rgba(0,0,0,0.4)] overflow-hidden">
                                <div className="px-4 py-3">
                                  <div className="text-xs font-medium text-zinc-500 mb-1">Assistant</div>
                        <div className="max-h-[25vh] overflow-y-auto">
                          <LaTeXMarkdown content={assistant.content} compact />
                        </div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  {expanded && (
                    <div className="mx-3 my-4 rounded-xl border border-zinc-800/40 bg-assistant-message shadow-sm overflow-hidden">
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
                      <div className="mx-5 mb-5 rounded-lg bg-zinc-900 border border-zinc-800/30 shadow-[0_-8px_24px_rgba(0,0,0,0.4)] overflow-hidden">
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
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </AnimatePresence>
          <div ref={bottomRef} />
          </div>
        </div>
        <div ref={inputAreaRef} className="absolute bottom-0 left-0 right-0 z-10 pointer-events-none">
          <div className="absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-zinc-950 via-zinc-950/60 to-transparent pointer-events-none" />
          <div className="relative pb-6 pointer-events-auto">
            <div className="mx-auto" style={chatMaxWidth ? { maxWidth: chatMaxWidth } : undefined}>
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
