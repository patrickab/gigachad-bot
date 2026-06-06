"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message, Attachment } from "@/lib/types"
import { ChatMessage } from "./ChatMessage"
import { ChatInput } from "./ChatInput"
import { ChatSidebar } from "./ChatSidebar"
import { getChatSidebarConfig } from "./chatSidebarConfig"
import { ChevronLeft, ChevronRight, User } from "lucide-react"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { cn } from "@/lib/utils"
import { ElevationProvider, ElevatedContainer } from "./ElevatedContainer"

const DEFAULT_EXPANDED_TAIL = 2
const BOTTOM_GAP_PX = 16 // Gap between last message and chat input, matching my-4 between QA pairs

interface ChatContainerProps {
  chatId: string
  messages: Message[]
  isStreaming: boolean
  onSend: (text: string, attachments: Attachment[]) => void
  onCancel: () => void
  onDeletePair: (index: number) => void
  className?: string
  slug?: string | null
  chatMaxWidth?: number
  onOCRRequest?: (image: string) => void
  onRemoveAttachment?: (messageIndex: number, attachmentName: string) => void
  onAttachmentContentChange?: (messageIndex: number, attachmentName: string, newContent: string) => void
}

export function ChatContainer({
  chatId,
  messages,
  isStreaming,
  onSend,
  onCancel,
  onDeletePair,
  className,
  slug = null,
  chatMaxWidth,
  onOCRRequest,
  onRemoveAttachment,
  onAttachmentContentChange,
}: ChatContainerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const isPinnedToBottomRef = useRef(true)

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
    const el = scrollContainerRef.current
    if (!el) return
    const handleScroll = () => {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
      isPinnedToBottomRef.current = nearBottom
    }
    el.addEventListener("scroll", handleScroll, { passive: true })
    return () => el.removeEventListener("scroll", handleScroll)
  }, [])

  useEffect(() => {
    if (!isPinnedToBottomRef.current) return
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

  const handleSend = useCallback((text: string, attachments: Attachment[]) => {
    isPinnedToBottomRef.current = true
    onSend(text, attachments)
  }, [onSend])

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
        slug,
        allAttachments,
        expandedEntries,
        onToggleAttachment: handleToggleExpand,
        onRemoveAttachment: handleRemoveAttachment,
        onAttachmentContentChange,
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
    [chatId, slug, allAttachments, expandedEntries, handleToggleExpand, handleRemoveAttachment, onAttachmentContentChange, lastMorphicResult, openElements]
  )

  const hasSidebarContent = sidebarElements.length > 0
  const inputAreaRef = useRef<HTMLDivElement>(null)
  const [inputAreaHeight, setInputAreaHeight] = useState(0)

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
    <div className={cn("flex h-full relative", className)}>
      <div className="flex-1 min-w-0 flex flex-col relative">
        <div ref={scrollContainerRef} className="flex-1 overflow-y-auto" style={{ paddingBottom: inputAreaHeight + BOTTOM_GAP_PX }}>
          <div className="mx-auto" style={{ maxWidth: chatMaxWidth || undefined }}>

          <ElevationProvider darkColor="var(--color-zinc-950)" brightColor="var(--color-zinc-900)" numLevels={3} startLevel={1}>
            <AnimatePresence>
            {pairs.map(({ user, assistant, globalIndex }) => {
              const expanded = isExpanded(globalIndex)
              const isLast = globalIndex === lastGlobalIndex
              const qLabel = abbreviate(user.content, user)
              const showPreview = hoveredPair === globalIndex
              return (
                <div key={globalIndex} className="group relative">
                  {!expanded && !isLast && (
                    <ElevatedContainer
                      onMouseEnter={() => handleMouseEnter(globalIndex)}
                      onMouseLeave={handleMouseLeave}
                      onClick={() => togglePair(globalIndex)}
                      className="mx-3 my-4 rounded-xl border border-zinc-800/40 shadow-sm overflow-hidden cursor-pointer transition-all duration-300"
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
                              <ElevatedContainer className="rounded-lg border border-zinc-800/30 overflow-hidden">
                                <div className="px-4 py-3">
                                  <div className="text-xs font-medium text-zinc-500 mb-1">Assistant</div>
                        <div className="max-h-[25vh] overflow-y-auto">
                          <LaTeXMarkdown content={assistant.content} compact />
                        </div>
                                  </div>
                              </ElevatedContainer>
                            )}
                          </div>
                        </div>
                      </div>
                    </ElevatedContainer>
                  )}
                  {expanded && (
                    <ElevatedContainer className="mx-3 my-4 rounded-xl border border-zinc-800/40 overflow-hidden">
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
                      <ElevatedContainer className="mx-5 mb-5 rounded-lg border border-zinc-800/30 overflow-hidden">
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
                      </ElevatedContainer>
                    </ElevatedContainer>
                  )}
                </div>
              )
            })}
          </AnimatePresence>
          </ElevationProvider>
          <div ref={bottomRef} />
          </div>
        </div>
        <div ref={inputAreaRef} className="absolute bottom-0 left-0 right-0 z-10 pointer-events-none">
          <div className="absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-zinc-950 via-zinc-950/60 to-transparent pointer-events-none" />
          <div className="relative pb-6 pointer-events-auto">
            <div className="mx-auto" style={chatMaxWidth ? { maxWidth: chatMaxWidth } : undefined}>
            <ChatInput
              chatId={chatId}
              onSend={handleSend}
              onOCRRequest={onOCRRequest}
              disabled={isStreaming}
              isStreaming={isStreaming}
              onCancel={onCancel}
              slug={slug}
            />
            </div>
          </div>
        </div>
        {hasSidebarContent && (
          <button
            onClick={() => setContextOpen((c) => !c)}
            title={contextOpen ? "Collapse sidebar" : "Open sidebar"}
            aria-label={contextOpen ? "Collapse sidebar" : "Open sidebar"}
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
