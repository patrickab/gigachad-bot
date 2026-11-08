"use client"

import { Fragment, useCallback, useMemo, useRef, useState } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message, Attachment } from "@/lib/domain/types"
import { ChatInput, type AttachMenuItem, type ChatContext, type ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { AttachmentPreview } from "@/features/chat/preview/AttachmentPreview"
import { ChevronsLeftRight, ChevronsRightLeft, GitFork } from "lucide-react"
import { ChatPair } from "./ChatPair"
import { findAttachment } from "@/features/chat/attachments"
import { usePairExpansion } from "@/features/chat/usePairExpansion"
import { usePairFocus } from "@/features/chat/usePairFocus"
import { useStickToBottom } from "@/features/chat/useStickToBottom"
import { useElementSize } from "@/hooks/useElementSize"
import { pairMessages } from "@/lib/domain/pairs"
import { InTopBar } from "@/features/navigation/TopBarSlots"

/** Chat column cap, omp-web's (`CHAT_COLUMN_MAX_WIDTH` in its `lib/chat-layout.ts`), so chat and omp tabs read alike. */
const CHAT_COLUMN_MAX_PX = 960
/** Extra scroll padding below the last pair, on top of the floating input height. */
const BOTTOM_GAP_PX = 16
/** An empty chat centers the composer this far above the column's middle, as a share of its height. */
const EMPTY_LIFT = 0.05
const EMPTY_GREETING = "What's on your mind?"

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
  /** The chat's saved history is still loading: its empty `messages` must not show the empty state. */
  historyLoading?: boolean
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
  /** Expose the message column for top-bar overlap measurement. */
  onChatColumnChange?: (element: HTMLDivElement | null) => void
}

export function ChatContainer({
  chatId,
  messages,
  historyLoading = false,
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
  onChatColumnChange,
}: ChatContainerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const columnRef = useRef<HTMLDivElement>(null)
  const inputAreaRef = useRef<HTMLDivElement>(null)
  const [isWideMode, setIsWideMode] = useState(false)
  // The chat whose first message was sent here: only that move to the bottom animates, a loading history jumps
  const [launchedChatId, setLaunchedChatId] = useState<string | null>(null)
  const [previewTarget, setPreviewTarget] = useState<{ messageIndex: number; name: string } | null>(null)
  const previewAttachment = previewTarget ? findAttachment(messages[previewTarget.messageIndex], previewTarget.name) ?? null : null

  const pairs = useMemo(() => pairMessages(messages), [messages])
  const { pin } = useStickToBottom(scrollContainerRef, bottomRef, chatId, messages, focus)
  const { isExpanded, toggle: togglePair } = usePairExpansion({ pairs, focus, messageCount: messages.length, scrollContainerRef })
  const { focusedGlobalIndex, clearKeyboardFocus, collapsedRef } = usePairFocus({ pairs, scrollContainerRef, togglePair })

  // Preserve composer spacing while hidden so returning tabs do not jump.
  const inputAreaSize = useElementSize(inputAreaRef)
  const [inputAreaHeight, setInputAreaHeight] = useState(0)
  if (inputAreaSize.height > 0 && inputAreaSize.height !== inputAreaHeight) setInputAreaHeight(inputAreaSize.height)
  const columnHeight = useElementSize(columnRef).height

  // A chat known to be empty lifts the composer from the bottom to slightly above the column's middle. A loading one keeps it at the bottom
  const isEmpty = messages.length === 0 && !historyLoading
  const emptyLift = isEmpty && columnHeight > 0 ? Math.max(0, columnHeight * (0.5 + EMPTY_LIFT) - inputAreaHeight / 2) : 0
  // The greeting waits for the measured, centered composer, so it never flashes at the bottom
  const showGreeting = emptyLift > 0 && inputAreaHeight > 0

  // Neovim starts from the text it is given, so a text preview waits for the fresh file. PDFs and images open at once and update.
  const onPreviewOpen = attachmentActions?.onPreviewOpen
  const openPreview = useCallback(async (messageIndex: number, attachment: Attachment) => {
    const reread = attachment.vaultPath ? onPreviewOpen?.(attachment.vaultPath) : undefined
    if (attachment.content != null) await reread
    setPreviewTarget({ messageIndex, name: attachment.name })
  }, [onPreviewOpen])

  // Sending re-pins to bottom so the new answer streams into view
  const handleSend = useCallback((text: string, attachments: Attachment[]) => {
    pin()
    if (messages.length === 0) setLaunchedChatId(chatId)
    onSend(text, attachments)
  }, [pin, onSend, messages.length, chatId])

  const context = useMemo<ChatContext | undefined>(
    () => attachmentActions && {
      attachments: messages.flatMap((m, messageIndex) => (m.attachments ?? []).map((attachment) => ({ messageIndex, attachment }))),
      onToggleActive: attachmentActions.onToggleActive,
      onRemove: attachmentActions.onRemove,
      onOpen: (messageIndex, attachment) => void openPreview(messageIndex, attachment),
    },
    [attachmentActions, messages, openPreview],
  )

  const effectiveMaxWidth = isWideMode ? undefined : CHAT_COLUMN_MAX_PX
  const toggleWideMode = useCallback(() => setIsWideMode((w) => !w), [])

  return (
    <div className="absolute inset-0 flex">
      <div ref={columnRef} className="flex-1 min-w-0 flex flex-col relative">
        <div
          ref={scrollContainerRef}
          tabIndex={-1}
          className="flex-1 overflow-y-auto text-ink outline-none"
          style={{ paddingBottom: inputAreaHeight + BOTTOM_GAP_PX }}
        >
          <div ref={onChatColumnChange} className="mx-auto" style={{ maxWidth: effectiveMaxWidth }}>
            <AnimatePresence>
            {pairs.map(({ user, assistant, globalIndex, qaIndex }) => (
              <Fragment key={globalIndex}>
                <ChatPair
                  user={user}
                  assistant={assistant}
                  globalIndex={globalIndex}
                  expanded={isExpanded(globalIndex)}
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
        <div
          ref={inputAreaRef}
          className={`absolute bottom-0 left-0 right-0 z-10 pointer-events-none ${launchedChatId === chatId ? "transition-transform duration-200 ease-out motion-reduce:transition-none" : ""}`}
          style={{ transform: `translateY(${-emptyLift}px)` }}
        >
          <p
            aria-hidden={!showGreeting}
            className={`absolute inset-x-0 bottom-full mb-5 text-center text-lg text-ink-subtle transition-opacity duration-200 ease-out motion-reduce:transition-none ${showGreeting ? "opacity-100" : "opacity-0"}`}
          >
            {EMPTY_GREETING}
          </p>
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
        <InTopBar name="wideToggle">
          <button
            type="button"
            onClick={toggleWideMode}
            aria-label={isWideMode ? "Exit wide mode" : "Enter wide mode"}
            className="rounded-lg p-1.5 text-ink-subtle opacity-0 transition-opacity duration-200 hover:text-ink hover:opacity-100 focus-visible:opacity-100"
          >
            {isWideMode ? <ChevronsRightLeft className="h-4 w-4" /> : <ChevronsLeftRight className="h-4 w-4" />}
          </button>
        </InTopBar>
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
