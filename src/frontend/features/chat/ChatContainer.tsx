"use client"

import { createPortal } from "react-dom"
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message, Attachment } from "@/lib/domain/types"
import { ChatInput, type ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { SidePanel } from "@/components/ui/SidePanel"
import { ChevronsLeftRight, ChevronsRightLeft, GitFork, PanelRight, PanelRightClose } from "lucide-react"
import "./chat.css"
import { ChatPair } from "./ChatPair"
import { useSidebarElements, type AttachmentEntry, type ExpandedAttachment } from "@/features/chat/context-panel/ChatContextPanel"
import { usePairFocus } from "@/features/chat/usePairFocus"
import { MAX_SIDEBAR_WIDTH } from "@/components/ui/SidePanel"

/** Trailing pairs expanded by default, older ones render collapsed. */
const DEFAULT_EXPANDED_TAIL = 1
/** Column width floor, matching the composer's max-w-2xl, so the chat is never narrower than the input. */
const MIN_CHAT_COLUMN_PX = 672
/** Extra scroll padding below the last pair, on top of the floating input height. */
const BOTTOM_GAP_PX = 16

/** Asks the chat to expand and scroll to one question; a new `key` re-fires the request for the same index. */
export interface ChatFocus {
  qaIndex: number | null
  key: number
}

interface ChatContainerProps {
  chatId: string
  messages: Message[]
  isStreaming: boolean
  onSend: (text: string, attachments: Attachment[]) => void
  onCancel: () => void
  onDeletePair: (index: number) => void
  onRegenerate?: (index: number) => void
  onBranch?: (qaIndex: number) => void
  focus?: ChatFocus
  branchMessageIdx?: number | null
  slug?: string | null
  extracting?: boolean
  chatInputRef?: React.RefObject<ChatInputHandle | null>
  /** Top-bar element the context-panel toggle renders into. */
  contextToggleSlot?: HTMLElement | null
  /** Top-bar element the wide-mode toggle renders into. */
  wideToggleSlot?: HTMLElement | null
  /** Expose the message column for top-bar overlap measurement. */
  onChatColumnChange?: (element: HTMLDivElement | null) => void
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
  focus,
  branchMessageIdx,
  slug = null,
  extracting,
  chatInputRef,
  contextToggleSlot,
  wideToggleSlot,
  onChatColumnChange,
}: ChatContainerProps) {
  const focusQaIndex = focus?.qaIndex ?? null
  const focusKey = focus?.key
  const bottomRef = useRef<HTMLDivElement>(null)
  const [editingDocPath, setEditingDocPath] = useState<string | null>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  // Ref, not state: updated on every scroll and read by the message effect without re-rendering
  const isPinnedToBottomRef = useRef(true)

  const lastSearchResult = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].search_result) return messages[i].search_result
    }
    return undefined
  }, [messages])

  const allAttachments = useMemo<AttachmentEntry[]>(() => {
    const seen = new Set<string>()
    const entries: AttachmentEntry[] = []
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
  const [expandedEntries, setExpandedEntries] = useState<ExpandedAttachment[]>([])
  const [sidebarWidth, setSidebarWidth] = useState(320)
  const [openElements, setOpenElements] = useState<Set<string>>(new Set())
  const [isWideMode, setIsWideMode] = useState(false)
  const [pdfWide, setPdfWide] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const savedSidebarWidth = useRef(320)

  // Widen the sidebar to half the container for PDFs, restoring the previous width on exit
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

  useEffect(() => {
    const el = scrollContainerRef.current
    if (!el) return
    const handleScroll = () => {
      // 80px slack so small overscrolls still count as pinned
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

  // Per-pair expand/collapse overrides keyed by global message index
  const [manualOverrides, setManualOverrides] = useState<Map<number, boolean>>(
    () => new Map(focusQaIndex == null ? [] : [[focusQaIndex * 2, true]]),
  )

  // A new focus request replaces the overrides: it expands only the focused pair
  const [seenFocus, setSeenFocus] = useState({ qaIndex: focusQaIndex, key: focusKey })
  if (seenFocus.qaIndex !== focusQaIndex || seenFocus.key !== focusKey) {
    setSeenFocus({ qaIndex: focusQaIndex, key: focusKey })
    setManualOverrides(new Map(focusQaIndex == null ? [] : [[focusQaIndex * 2, true]]))
  }

  // Depend on the length, not the array: in an empty chat every re-render hands
  // down a fresh [] and this would slam the sidebar shut again, including right
  // after you opened the Documents expander.
  const [seenMessageCount, setSeenMessageCount] = useState(messages.length)
  if (seenMessageCount !== messages.length) {
    setSeenMessageCount(messages.length)
    if (messages.length === 0) {
      setManualOverrides(new Map())
      setOpenElements(new Set())
    }
  }

  // Scroll to the pair before the focused one so the target sits just below a collapsed header
  useEffect(() => {
    if (focusQaIndex == null) return
    const gi = focusQaIndex * 2
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

  // Pair user/assistant messages, padding a missing assistant reply with an empty one
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

  // Pairs from this index on are expanded unless overridden
  const tailStartGlobalIndex = pairs.length > DEFAULT_EXPANDED_TAIL
    ? pairs[pairs.length - DEFAULT_EXPANDED_TAIL].globalIndex
    : (pairs.length > 0 ? pairs[0].globalIndex : -1)

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

  const { focusedGlobalIndex, clearKeyboardFocus, collapsedRef } = usePairFocus({ pairs, scrollContainerRef, togglePair })

  // Sending re-pins to bottom so the new answer streams into view
  const handleSend = useCallback((text: string, attachments: Attachment[]) => {
    isPinnedToBottomRef.current = true
    onSend(text, attachments)
  }, [onSend])

  const isElementOpen = useCallback((id: string) => openElements.has(id), [openElements])
  const onElementOpenChange = useCallback((id: string, open: boolean) => {
    setOpenElements((prev) => {
      const next = new Set(prev)
      if (open) next.add(id)
      else next.delete(id)
      return next
    })
  }, [])

  const sidebarElements = useSidebarElements({
    slug,
    allAttachments,
    expandedEntries,
    onToggleExpand: handleToggleExpand,
    lastSearchResult,
    editingDocPath,
    onEditDocument: setEditingDocPath,
    isElementOpen,
    onElementOpenChange,
    pdfWide,
    onTogglePdfWide: togglePdfWide,
  })

  const hasSidebarContent = sidebarElements.length > 0
  const inputAreaRef = useRef<HTMLDivElement>(null)
  const [inputAreaHeight, setInputAreaHeight] = useState(0)

  // Preserve composer spacing while hidden so returning tabs do not jump.
  useEffect(() => {
    const el = inputAreaRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.height > 0) setInputAreaHeight(entry.contentRect.height)
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Nothing left to show, or leaving the context panel, drops the panel and PDF wide mode
  if (!hasSidebarContent && contextOpen) setContextOpen(false)
  if (!contextOpen && pdfWide) setPdfWide(false)

  // Match the collapsed-sidebar width, shrinking only when the available column cannot fit it.
  const effectiveMaxWidth = isWideMode ? undefined : `max(${MIN_CHAT_COLUMN_PX}px, calc(100vw - ${MAX_SIDEBAR_WIDTH}px))`
  const toggleWideMode = useCallback(() => setIsWideMode((w) => !w), [])

  return (
    <div ref={containerRef} className="absolute inset-x-0 -top-11 bottom-0 flex">
      <div className="flex-1 min-w-0 flex flex-col relative">
        <div
          ref={scrollContainerRef}
          tabIndex={-1}
          className="flex-1 overflow-y-auto text-ink outline-none"
          style={{ paddingBottom: inputAreaHeight + BOTTOM_GAP_PX }}
        >
          <div ref={onChatColumnChange} className="mx-auto" style={{ maxWidth: effectiveMaxWidth }}>
            <AnimatePresence>
            {pairs.map(({ user, assistant, globalIndex }, qaIndex) => (
              <Fragment key={globalIndex}>
                <ChatPair
                  user={user}
                  assistant={assistant}
                  globalIndex={globalIndex}
                  expanded={isExpandedNormal(globalIndex)}
                  focused={focusedGlobalIndex === globalIndex}
                  isStreaming={isStreaming}
                  onExpand={() => {
                    togglePair(globalIndex)
                    clearKeyboardFocus()
                  }}
                  onCollapse={() => togglePair(globalIndex)}
                  collapsedRef={collapsedRef(globalIndex)}
                  onAttachmentClick={(att) => handleAttachmentClick(globalIndex, att)}
                  onDelete={onDeletePair}
                  onRegenerate={onRegenerate}
                  onBranch={onBranch ? () => onBranch(qaIndex) : undefined}
                />
                {branchMessageIdx != null && qaIndex === branchMessageIdx && (
                  <div role="separator" className="flex items-center gap-2 mx-3 my-3">
                    <GitFork className="h-3.5 w-3.5 shrink-0 text-ink-subtle" aria-hidden="true" />
                    <span className="text-[10px] font-medium text-ink-muted tabular-nums">branched@{branchMessageIdx}</span>
                    <div className="flex-1 h-px bg-divider-strong" />
                  </div>
                )}
              </Fragment>
            ))}
          </AnimatePresence>
          <div ref={bottomRef} />
          </div>
        </div>
        <div ref={inputAreaRef} className="absolute bottom-0 left-0 right-0 z-10 pointer-events-none">
          <div className="absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-paper via-paper/40 via-30% to-transparent pointer-events-none" />
          <div className="relative px-4 pb-6 pointer-events-auto">
            <ChatInput
              ref={chatInputRef}
              chatId={chatId}
              onSend={handleSend}
              disabled={isStreaming || extracting}
              extracting={extracting}
              isStreaming={isStreaming}
              onCancel={onCancel}
              slug={slug}
            />
          </div>
        </div>
        {hasSidebarContent && contextToggleSlot && createPortal(
          <button
            type="button"
            onClick={() => setContextOpen((c) => !c)}
            aria-label={contextOpen ? "Close context panel" : "Open context panel"}
            aria-expanded={contextOpen}
            className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
          >
            {contextOpen ? <PanelRightClose className="h-4 w-4" /> : <PanelRight className="h-4 w-4" />}
          </button>,
          contextToggleSlot,
        )}
        {wideToggleSlot && createPortal(
          <button
            type="button"
            onClick={toggleWideMode}
            aria-label={isWideMode ? "Exit wide mode" : "Enter wide mode"}
            className="rounded-lg p-1.5 text-ink-subtle opacity-0 transition-opacity duration-200 hover:text-ink hover:opacity-100 focus-visible:opacity-100"
          >
            {isWideMode ? <ChevronsRightLeft className="h-4 w-4" /> : <ChevronsLeftRight className="h-4 w-4" />}
          </button>,
          wideToggleSlot,
        )}
      </div>
      {hasSidebarContent && contextOpen && chatId && (
        <SidePanel sections={sidebarElements} width={sidebarWidth} onWidthChange={setSidebarWidth} maxWidth={pdfWide ? sidebarWidth : undefined} />
      )}
    </div>
  )
}
