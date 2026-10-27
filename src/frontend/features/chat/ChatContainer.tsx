"use client"

import { createPortal } from "react-dom"
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message, Attachment } from "@/lib/domain/types"
import { ChatInput, type AttachMenuItem, type ChatContext, type ChatInputHandle } from "@/features/chat/composer/ChatInput"
import type { ContextAttachment } from "@/features/chat/composer/AttachmentChip"
import { AttachmentPreview } from "@/features/chat/preview/AttachmentPreview"
import { ChevronsLeftRight, ChevronsRightLeft, GitFork } from "lucide-react"
import { ChatPair } from "./ChatPair"
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

/** What the chat does with its attachments: composer chips and the previews of message attachments. */
export interface AttachmentActions {
  onToggleActive: (messageIndex: number, attachmentName: string) => void
  onRemove: (messageIndex: number, attachmentName: string) => void
  onContentChange: (messageIndex: number, attachmentName: string, newContent: string) => void | Promise<void>
  /** Re-read the reference `path`, so its preview shows the file as it is now. */
  onPreviewOpen: (path: string) => Promise<void>
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
  /** Context actions of the chat's attachments (composer chips and previews). */
  attachmentActions?: AttachmentActions
  /** Extra entries of the composer's "+" menu. */
  attachMenuItems?: AttachMenuItem[]
  /** Note under the draft, e.g. why the last attach failed. */
  composerNotice?: string
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
  attachmentActions,
  attachMenuItems,
  composerNotice,
  wideToggleSlot,
  onChatColumnChange,
}: ChatContainerProps) {
  const focusQaIndex = focus?.qaIndex ?? null
  const focusKey = focus?.key
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  // Ref, not state: updated on every scroll and read by the message effect without re-rendering
  const isPinnedToBottomRef = useRef(true)
  const previousMessageCount = useRef(0)

  const contextAttachments = useMemo<ContextAttachment[]>(() => {
    const seen = new Set<string>()
    const entries: ContextAttachment[] = []
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

  const [isWideMode, setIsWideMode] = useState(false)
  const [previewTarget, setPreviewTarget] = useState<{ messageIndex: number; name: string } | null>(null)
  const previewAttachment = previewTarget
    ? messages[previewTarget.messageIndex]?.attachments?.find((a) => a.name === previewTarget.name) ?? null
    : null

  // Neovim starts from the text it is given, so a text preview waits for the fresh file. PDFs and images open at once and update.
  const onPreviewOpen = attachmentActions?.onPreviewOpen
  const openPreview = useCallback(async (messageIndex: number, attachment: Attachment) => {
    const reread = attachment.vaultPath ? onPreviewOpen?.(attachment.vaultPath) : undefined
    if (attachment.content != null) await reread
    setPreviewTarget({ messageIndex, name: attachment.name })
  }, [onPreviewOpen])

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
    const wasEmpty = previousMessageCount.current === 0
    previousMessageCount.current = messages.length
    if (!isPinnedToBottomRef.current) return
    // Restored histories jump straight to the end, avoiding work in every preview along a smooth scroll.
    bottomRef.current?.scrollIntoView({ behavior: wasEmpty ? "instant" : "smooth" })
  }, [messages])

  // Per-pair expand/collapse overrides keyed by global message index
  const focusOverrides = () => new Map<number, boolean>(focusQaIndex == null ? [] : [[focusQaIndex * 2, true]])
  const [manualOverrides, setManualOverrides] = useState(focusOverrides)

  // A new focus request replaces the overrides: it expands only the focused pair
  const [seenFocus, setSeenFocus] = useState({ qaIndex: focusQaIndex, key: focusKey })
  if (seenFocus.qaIndex !== focusQaIndex || seenFocus.key !== focusKey) {
    setSeenFocus({ qaIndex: focusQaIndex, key: focusKey })
    setManualOverrides(focusOverrides())
  }

  // Depend on the length, not the array: in an empty chat every re-render hands
  // down a fresh [] and would reset the overrides each time.
  const [seenMessageCount, setSeenMessageCount] = useState(messages.length)
  if (seenMessageCount !== messages.length) {
    setSeenMessageCount(messages.length)
    if (messages.length === 0) setManualOverrides(new Map())
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

  const context = useMemo<ChatContext | undefined>(
    () => attachmentActions && {
      attachments: contextAttachments,
      onToggleActive: attachmentActions.onToggleActive,
      onRemove: attachmentActions.onRemove,
      onOpen: (messageIndex, attachment) => void openPreview(messageIndex, attachment),
    },
    [attachmentActions, contextAttachments, openPreview],
  )

  // Match the collapsed-sidebar width, shrinking only when the available column cannot fit it.
  const effectiveMaxWidth = isWideMode ? undefined : `max(${MIN_CHAT_COLUMN_PX}px, calc(100vw - ${MAX_SIDEBAR_WIDTH}px))`
  const toggleWideMode = useCallback(() => setIsWideMode((w) => !w), [])

  return (
    <div className="absolute inset-0 flex">
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
                  onAttachmentClick={(att) => void openPreview(globalIndex, att)}
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
              context={context}
              attachMenuItems={attachMenuItems}
              hint={composerNotice ? () => composerNotice : undefined}
            />
          </div>
        </div>
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
      <AttachmentPreview
        attachment={previewAttachment}
        onClose={() => setPreviewTarget(null)}
        onContentChange={previewTarget && attachmentActions
          ? (newContent) => attachmentActions.onContentChange(previewTarget.messageIndex, previewTarget.name, newContent)
          : undefined}
      />
    </div>
  )
}
