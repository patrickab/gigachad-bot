"use client"

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message, Attachment, ProjectDocument } from "@/lib/types"
import { ChatMessage, AssistantMessageContent } from "./ChatMessage"
import { ChatInput } from "./ChatInput"
import { ChatSidebar } from "./ChatSidebar"
import { getChatSidebarConfig } from "./chatSidebarConfig"

import { ChevronLeft, ChevronRight, ChevronsLeftRight, ChevronsRightLeft, GitFork, User } from "lucide-react"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { cn } from "@/lib/utils"
import { ElevationProvider, ElevatedContainer } from "./ElevatedContainer"

const DEFAULT_EXPANDED_TAIL = 1
const BOTTOM_GAP_PX = 16

interface ChatContainerProps {
  chatId: string
  messages: Message[]
  isStreaming: boolean
  onSend: (text: string, attachments: Attachment[]) => void
  onCancel: () => void
  onDeletePair: (index: number) => void
  onRegenerate?: (index: number) => void
  onBranch?: (qaIndex: number) => void
  focusQaIndex?: number | null
  focusKey?: number
  branchMessageIdx?: number | null
  isActive?: boolean
  className?: string
  slug?: string | null
  chatMaxWidth?: number
  onOCRRequest?: (image: string) => void
  onRemoveAttachment?: (messageIndex: number, attachmentName: string) => void
  onToggleAttachmentActive?: (messageIndex: number, attachmentName: string) => void
  onAttachmentContentChange?: (messageIndex: number, attachmentName: string, newContent: string) => void
  obsidianEnabled?: boolean
  onOpenObsidian?: () => void
  documents?: ProjectDocument[]
  onSelectDocument?: (path: string) => void
  onOpenDocuments?: () => void
}

export function ChatContainer({
  chatId,
  messages,
  isStreaming,
  onSend,
  onCancel,
  onDeletePair,
  onRegenerate,
  onBranch,
  focusQaIndex,
  focusKey,
  branchMessageIdx,
  isActive,
  className,
  slug = null,
  chatMaxWidth,
  onOCRRequest,
  onRemoveAttachment,
  onToggleAttachmentActive,
  onAttachmentContentChange,
  obsidianEnabled,
  onOpenObsidian,
  documents,
  onSelectDocument,
  onOpenDocuments,
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
  const [isWideMode, setIsWideMode] = useState(false)
  const [pdfWide, setPdfWide] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const savedSidebarWidth = useRef(320)

  const togglePdfWide = useCallback(() => {
    setPdfWide((prev) => {
      if (!prev) {
        savedSidebarWidth.current = sidebarWidth
        const total = containerRef.current?.clientWidth ?? 0
        if (total > 0) setSidebarWidth(Math.round(total / 2))
      } else {
        setSidebarWidth(savedSidebarWidth.current)
      }
      return !prev
    })
  }, [sidebarWidth])

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
    if (focusQaIndex == null) {
      setManualOverrides(new Map())
      return
    }
    const gi = focusQaIndex * 2
    setManualOverrides(new Map([[gi, true]]))
    requestAnimationFrame(() => {
      const el = scrollContainerRef.current
      if (!el) return
      const anchorGi = gi - 2
      const target = anchorGi >= 0
        ? el.querySelector(`[data-qa-index="${anchorGi}"]`)
        : el.querySelector(`[data-qa-index="${gi}"]`)
      target?.scrollIntoView({ behavior: "smooth", block: "start" })
    })
  }, [focusQaIndex, focusKey])

  useEffect(() => {
    if (messages.length === 0) {
      setManualOverrides(new Map())
      setOpenElements(new Set())
    }
  }, [messages])

  const pairs = useMemo<{ user: Message; assistant: Message; globalIndex: number }[]>(() => {
    const out: { user: Message; assistant: Message; globalIndex: number }[] = []
    for (let i = 0; i < messages.length; i += 2) {
      if (messages[i]?.role === "user") {
        out.push({
          user: messages[i],
          assistant: messages[i + 1] ?? { role: "assistant", content: "" },
          globalIndex: i,
        })
      }
    }
    return out
  }, [messages])

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

  function isExpandedNormal(idx: number): boolean {
    const override = manualOverrides.get(idx)
    if (override !== undefined) return override
    return idx >= tailStartGlobalIndex
  }

  function togglePair(idx: number) {
    setManualOverrides(prev => {
      const next = new Map(prev)
      const current = isExpandedNormal(idx)
      next.set(idx, !current)
      return next
    })
  }

  const [hoveredPair, setHoveredPair] = useState<number | null>(null)
  const [keyboardFocusIdx, setKeyboardFocusIdx] = useState<number | null>(null)
  const collapsedRefs = useRef<Map<number, HTMLDivElement>>(new Map())
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const leaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const allPairsList = useMemo<number[]>(() => {
    const gs: number[] = []
    for (let i = 0; i < messages.length; i += 2) {
      if (messages[i]?.role === "user") {
        gs.push(i)
      }
    }
    return gs
  }, [messages])

  const activeFocusedGlobalIndex =
    keyboardFocusIdx !== null && keyboardFocusIdx >= 0 && keyboardFocusIdx < allPairsList.length
      ? allPairsList[keyboardFocusIdx]
      : null

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
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
    if (leaveTimeoutRef.current) clearTimeout(leaveTimeoutRef.current)
    leaveTimeoutRef.current = setTimeout(() => {
      setHoveredPair(null)
      leaveTimeoutRef.current = null
    }, 100)
  }

  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
      if (leaveTimeoutRef.current) clearTimeout(leaveTimeoutRef.current)
    }
  }, [])

  const navigateFocus = useCallback((key: "ArrowUp" | "ArrowDown") => {
    if (allPairsList.length === 0) {
      setKeyboardFocusIdx(null)
      return
    }
    setKeyboardFocusIdx((cur) => {
      const start = cur ?? (key === "ArrowDown" ? -1 : allPairsList.length)
      return Math.max(0, Math.min(start + (key === "ArrowDown" ? 1 : -1), allPairsList.length - 1))
    })
  }, [allPairsList])

  const confirmFocus = useCallback(() => {
    if (keyboardFocusIdx === null || keyboardFocusIdx < 0 || keyboardFocusIdx >= allPairsList.length) return
    togglePair(allPairsList[keyboardFocusIdx])
    setKeyboardFocusIdx(null)
  }, [keyboardFocusIdx, allPairsList, togglePair])

  useEffect(() => {
    const eat = (e: KeyboardEvent) => { e.preventDefault(); e.stopPropagation() }
    const handleKeyDown = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey
      if (ctrl && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
        eat(e)
        navigateFocus(e.key)
        scrollContainerRef.current?.focus()
      } else if (e.key === "Enter" && keyboardFocusIdx !== null) {
        eat(e)
        confirmFocus()
      } else if (e.key === "Escape" && keyboardFocusIdx !== null) {
        eat(e)
        setKeyboardFocusIdx(null)
      }
    }
    window.addEventListener("keydown", handleKeyDown, true)
    return () => window.removeEventListener("keydown", handleKeyDown, true)
  }, [navigateFocus, confirmFocus, keyboardFocusIdx])

  useEffect(() => {
    if (keyboardFocusIdx === null) return
    const container = scrollContainerRef.current
    if (!container) return

    const atTop = keyboardFocusIdx < 3
    const targetIdx = atTop ? 0 : keyboardFocusIdx - 3
    const targetGlobalIdx = allPairsList[targetIdx]
    if (targetGlobalIdx === undefined) return
    const el = collapsedRefs.current.get(targetGlobalIdx)
    if (!el) return

    const elRect = el.getBoundingClientRect()
    const containerRect = container.getBoundingClientRect()
    const offset = (atTop ? elRect.top : elRect.bottom) - containerRect.top

    container.scrollTo({
      top: container.scrollTop + offset,
      behavior: "smooth"
    })
  }, [keyboardFocusIdx, allPairsList])

  const sidebarElements = useMemo(
    () =>
      getChatSidebarConfig({
        chatId,
        slug,
        allAttachments,
        expandedEntries,
        onToggleExpand: handleToggleExpand,
        onToggleAttachmentActive,
        onRemoveAttachment: handleRemoveAttachment,
        onAttachmentContentChange,
        lastMorphicResult,
        obsidianEnabled,
        onOpenObsidian,
        documents,
        onSelectDocument,
        onOpenDocuments,
        isElementOpen: (id) => openElements.has(id),
        onElementOpenChange: (id, open) => {
          setOpenElements((prev) => {
            const next = new Set(prev)
            if (open) next.add(id)
            else next.delete(id)
            return next
          })
        },
        pdfWide,
        onTogglePdfWide: togglePdfWide,
      }),
    [chatId, slug, allAttachments, expandedEntries, handleToggleExpand, onToggleAttachmentActive, handleRemoveAttachment, onAttachmentContentChange, lastMorphicResult, obsidianEnabled, onOpenObsidian, documents, onSelectDocument, onOpenDocuments, openElements, pdfWide, togglePdfWide]
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

  useEffect(() => {
    if (!contextOpen) setPdfWide(false)
  }, [contextOpen])

  const effectiveMaxWidth = isWideMode ? undefined : (chatMaxWidth || undefined)
  const toggleWideMode = useCallback(() => setIsWideMode((w) => !w), [])

  return (
    <div ref={containerRef} className={cn("flex h-full relative", className)}>
      <div className="flex-1 min-w-0 flex flex-col relative">
        <div
          ref={scrollContainerRef}
          tabIndex={-1}
          className="flex-1 overflow-y-auto text-ink outline-none"
          style={{ paddingBottom: inputAreaHeight + BOTTOM_GAP_PX }}
        >
          <div className="mx-auto" style={{ maxWidth: effectiveMaxWidth }}>
          <button
            onClick={toggleWideMode}
            aria-label={isWideMode ? "Exit wide mode" : "Enter wide mode"}
            className="absolute left-2 top-2 z-30 p-1 text-ink-faint opacity-0 hover:opacity-100 focus-visible:opacity-100 hover:text-ink-muted transition-opacity duration-200 cursor-pointer"
          >
            {isWideMode ? <ChevronsRightLeft className="h-4 w-4" /> : <ChevronsLeftRight className="h-4 w-4" />}
          </button>
          <ElevationProvider darkColor="var(--paper)" brightColor="var(--surface-elevated)" numLevels={3} startLevel={1}>
            <AnimatePresence>
            {pairs.map(({ user, assistant, globalIndex }, qaIndex) => {
              const expanded = isExpandedNormal(globalIndex)
              const qLabel = abbreviate(user.content, user)
              const showPreview = hoveredPair === globalIndex || activeFocusedGlobalIndex === globalIndex
              const showBranchDivider = branchMessageIdx != null && qaIndex === branchMessageIdx
              return (
                <Fragment key={globalIndex}>
                <div data-qa-index={globalIndex} className="group relative">
                  {!expanded && (
                    <ElevatedContainer
                      ref={(el) => {
                        if (el) collapsedRefs.current.set(globalIndex, el)
                        else collapsedRefs.current.delete(globalIndex)
                      }}
                      onMouseEnter={() => handleMouseEnter(globalIndex)}
                      onMouseLeave={handleMouseLeave}
                      onClick={() => {
                        togglePair(globalIndex)
                        setKeyboardFocusIdx(null)
                      }}
                      className={cn(
                        "mx-3 my-4 rounded-xl border border-divider/40 shadow-[var(--shadow-sm)] overflow-hidden cursor-pointer transition-all duration-300",
                        activeFocusedGlobalIndex === globalIndex && "ring-1 ring-ink-muted/50"
                      )}
                    >
                      <div className="flex gap-3 px-5 py-4 items-start">
                        <div className="mt-0.5 shrink-0">
                          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-surface-elevated">
                            <User className="h-3.5 w-3.5 text-ink-muted" />
                          </div>
                        </div>
                        <div className="min-w-0 flex-1 flex flex-col">
                          <div className="mb-0.5 text-xs font-medium text-ink-subtle">You</div>
                          {showPreview ? (
                            <div className="max-h-[12.5vh] overflow-y-auto text-ink-muted">
                              <LaTeXMarkdown content={user.content} compact />
                            </div>
                          ) : (
                            <span className="text-sm text-ink-muted truncate">{qLabel}</span>
                          )}
                        </div>
                        <ChevronRight className="h-4 w-4 shrink-0 text-ink-faint mt-1" />
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
                              <ElevatedContainer className="rounded-lg border border-divider/30 overflow-hidden">
                                <div className="px-4 py-3">
                                  <div className="text-xs font-medium text-ink-subtle mb-1">Assistant</div>
                        <div className="max-h-[25vh] overflow-y-auto text-ink-muted">
                          <AssistantMessageContent content={assistant.content} compact />
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
                    <>

                      <ElevatedContainer className={cn("mx-3 my-4 rounded-xl border border-divider/40 overflow-hidden transition-all duration-300", activeFocusedGlobalIndex === globalIndex && "ring-1 ring-ink-muted/50")}>
                        <ChatMessage
                          role="user"
                          content={user.content}
                          attachments={user.attachments}
                          index={globalIndex}
                          onAttachmentClick={(att) => handleAttachmentClick(globalIndex, att)}
                          onDelete={onDeletePair}
                          collapsibleUser
                          onCollapse={() => togglePair(globalIndex)}
                        />
                        <ElevatedContainer className="mx-5 mb-5 rounded-lg border border-divider/30 overflow-hidden">
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
                            onRegenerate={onRegenerate}
                             onBranch={onBranch ? () => onBranch(qaIndex) : undefined}
                          />
                        </ElevatedContainer>
                      </ElevatedContainer>
                    </>
                  )}
                </div>
                {showBranchDivider && (
                  <div role="separator" className="flex items-center gap-2 mx-3 my-3">
                    <GitFork className="h-3.5 w-3.5 shrink-0 text-ink-subtle" aria-hidden="true" />
                    <span className="text-[10px] font-medium text-ink-muted tabular-nums">branched@{branchMessageIdx}</span>
                    <div className="flex-1 h-px bg-divider-strong" />
                  </div>
                )}
                </Fragment>
              )
            })}
          </AnimatePresence>
          </ElevationProvider>
          <div ref={bottomRef} />
          </div>
        </div>
        <div ref={inputAreaRef} className="absolute bottom-0 left-0 right-0 z-10 pointer-events-none">
          <div className="absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-paper via-paper/40 via-30% to-transparent pointer-events-none" />
          <div className="relative pb-6 pointer-events-auto">
            <div className="mx-auto" style={chatMaxWidth && !isWideMode ? { maxWidth: chatMaxWidth } : undefined}>
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
            aria-expanded={contextOpen}
            className="absolute right-0 top-3 z-30 flex items-center justify-center h-12 w-4 rounded-l-md border border-r-0 border-divider-strong bg-surface-elevated/80 text-ink hover:bg-surface-elevated hover:text-ink hover:w-5 transition-all shadow-[var(--shadow-sm)]"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        )}
      </div>
      {hasSidebarContent && contextOpen && chatId && (
        <ChatSidebar elements={sidebarElements} width={sidebarWidth} onWidthChange={setSidebarWidth} maxWidth={pdfWide ? sidebarWidth : undefined} />
      )}
    </div>
  )
}
