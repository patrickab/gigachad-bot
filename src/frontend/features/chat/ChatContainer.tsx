"use client"

import { CHAT_COLUMN_MAX_PX } from "@/components/ui/columnResize"
import { useCallback, useMemo, useState, type ReactNode } from "react"
import type { Message, Attachment } from "@/lib/domain/types"
import { ChatInput, type AttachMenuItem, type ChatContext, type ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { AttachmentPreview } from "@/features/chat/preview/AttachmentPreview"
import { GitFork } from "lucide-react"
import { ConversationColumn, type ConversationReveal, type ConversationTurn } from "@/features/conversation/ConversationColumn"
import { ChatPairBody, ChatPairPreview, pairLabel } from "./ChatPair"
import { findAttachment } from "@/features/chat/attachments"
import { keyPairs, pairMessages, type Pair } from "@/lib/domain/pairs"
import { ComposerDock, useComposerDock } from "./ComposerDock"

/** Max width of a chat's Q&A column outside wide mode, shared with the Pi transcript. */
export { CHAT_COLUMN_MAX_PX }

/** Asks the chat to open and scroll to one question, the column's `reveal` by pair; a new `seq` re-fires it for the same index. */
export interface ChatReveal {
  qaIndex: number | null
  seq: number
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
  /** Conversation the composer keeps its unsent text for, see `ChatInput`. */
  draftKey?: string | null
  messages: Message[]
  /** The chat's saved history is still loading: its empty `messages` must not show the empty state. */
  historyLoading?: boolean
  isStreaming: boolean
  onSend: (text: string, attachments: Attachment[]) => void
  onCancel: () => void
  onDeletePair: (index: number) => void
  onRegenerate?: (index: number) => void
  onBranch?: (qaIndex: number) => void
  reveal?: ChatReveal
  branchMessageIdx?: number | null
  slug?: string | null
  extracting?: boolean
  chatInputRef?: React.RefObject<ChatInputHandle | null>
  /** Context actions of the chat's attachments (composer chips and previews). */
  attachmentActions?: AttachmentActions
  /** Extra entries of the composer's "+" menu. */
  attachMenuItems?: AttachMenuItem[]
  /** Extra controls of the composer's action row, after the "+" menu. */
  composerLeading?: ReactNode
  /** Note under the draft, e.g. why the last attach failed. */
  composerNotice?: string
  /** Expose the message column for top-bar overlap measurement. */
  onChatColumnChange?: (element: HTMLDivElement | null) => void
}

/** A pair as the column's turn. */
interface ChatTurn extends ConversationTurn {
  pair: Pair
}

/** Keys of `pairs` that survive the delete of an earlier pair, see `keyPairs`. */
function usePairKeys(pairs: Pair[]): string[] {
  const [state, setState] = useState(() => ({ pairs, ...keyPairs(pairs) }))
  if (state.pairs === pairs) return state.keys
  const next = { pairs, ...keyPairs(pairs, state) }
  setState(next)
  return next.keys
}

/** Where a branched chat left its parent: after the pair it branched at. */
function BranchSeparator({ at }: { at: number }) {
  return (
    <div role="separator" className="flex items-center gap-2 mx-3 my-3">
      <GitFork className="h-3.5 w-3.5 shrink-0 text-ink-subtle" aria-hidden="true" />
      <span className="text-[10px] font-medium text-ink-muted tabular-nums">branched@{at}</span>
      <div className="flex-1 h-px bg-divider-strong" />
    </div>
  )
}

export function ChatContainer({
  chatId,
  draftKey,
  messages,
  historyLoading = false,
  isStreaming,
  onSend,
  onCancel,
  onDeletePair,
  onRegenerate,
  onBranch,
  reveal,
  branchMessageIdx,
  slug = null,
  extracting,
  chatInputRef,
  attachmentActions,
  attachMenuItems,
  composerLeading,
  composerNotice,
  onChatColumnChange,
}: ChatContainerProps) {
  // The chat whose first message was sent here: only that move to the bottom animates, a loading history jumps
  const [launchedChatId, setLaunchedChatId] = useState<string | null>(null)
  const [previewTarget, setPreviewTarget] = useState<{ messageIndex: number; name: string } | null>(null)
  const previewAttachment = previewTarget ? findAttachment(messages[previewTarget.messageIndex], previewTarget.name) ?? null : null
  // Bumped on send: the column follows the end again
  const [followSeq, setFollowSeq] = useState(0)

  const pairs = useMemo(() => pairMessages(messages), [messages])
  const keys = usePairKeys(pairs)
  const turns = useMemo(() => pairs.map((pair, index): ChatTurn => ({
    key: keys[index],
    label: pairLabel(pair.user),
    live: isStreaming && index === pairs.length - 1,
    pair,
  })), [pairs, keys, isStreaming])
  const revealQa = reveal?.qaIndex ?? null
  const revealKey = revealQa === null ? undefined : turns[revealQa]?.key
  const revealSeq = reveal?.seq ?? 0
  // Waits for the pair: a chat opened at a question loads its history first
  const columnReveal = useMemo<ConversationReveal | null>(() => (revealKey === undefined ? null : { key: revealKey, seq: revealSeq }), [revealKey, revealSeq])

  // A chat known to be empty lifts the composer to just below the column's middle. A loading one keeps it at the bottom
  const dock = useComposerDock(messages.length === 0 && !historyLoading)
  const { mountColumn } = dock

  // Neovim starts from the text it is given, so a text preview waits for the fresh file. PDFs and images open at once and update.
  const onPreviewOpen = attachmentActions?.onPreviewOpen
  const openPreview = useCallback(async (messageIndex: number, attachment: Attachment) => {
    const reread = attachment.vaultPath ? onPreviewOpen?.(attachment.vaultPath) : undefined
    if (attachment.content != null) await reread
    setPreviewTarget({ messageIndex, name: attachment.name })
  }, [onPreviewOpen])

  // Sending re-pins to bottom so the new answer streams into view
  const handleSend = useCallback((text: string, attachments: Attachment[]) => {
    setFollowSeq((seq) => seq + 1)
    if (messages.length === 0) setLaunchedChatId(chatId)
    onSend(text, attachments)
  }, [onSend, messages.length, chatId])

  const context = useMemo<ChatContext | undefined>(
    () => attachmentActions && {
      attachments: messages.flatMap((m, messageIndex) => (m.attachments ?? []).map((attachment) => ({ messageIndex, attachment }))),
      onToggleActive: attachmentActions.onToggleActive,
      onRemove: attachmentActions.onRemove,
      onOpen: (messageIndex, attachment) => void openPreview(messageIndex, attachment),
    },
    [attachmentActions, messages, openPreview],
  )

  // Settled pairs get equal props each streamed frame, so the column skips them
  const renderPreview = (turn: ChatTurn) => <ChatPairPreview assistant={turn.pair.assistant} />
  const renderBody = (turn: ChatTurn, _index: number, collapse: (() => void) | undefined) => (
    <ChatPairBody
      user={turn.pair.user}
      assistant={turn.pair.assistant}
      globalIndex={turn.pair.globalIndex}
      qaIndex={turn.pair.qaIndex}
      isStreaming={isStreaming}
      onCollapse={collapse}
      onAttachmentClick={openPreview}
      onDelete={onDeletePair}
      onRegenerate={onRegenerate}
      onBranch={onBranch}
    />
  )
  const renderAfter = (turn: ChatTurn) => (branchMessageIdx != null && turn.pair.qaIndex === branchMessageIdx ? <BranchSeparator at={branchMessageIdx} /> : null)

  return (
    <div className="absolute inset-0 flex">
      <div ref={mountColumn} className="flex-1 min-w-0 flex flex-col relative">
        {/* Another chat opens at its end again */}
        <ConversationColumn
          key={chatId}
          turns={turns}
          renderPreview={renderPreview}
          renderBody={renderBody}
          renderAfter={renderAfter}
          reveal={columnReveal}
          followSeq={followSeq}
          paddingBottom={dock.paddingBottom}
          onColumnChange={onChatColumnChange}
        />
        <ComposerDock dock={dock} animate={launchedChatId === chatId}>
          <ChatInput
            ref={chatInputRef}
            chatId={chatId}
            draftKey={draftKey}
            onSend={handleSend}
            disabled={isStreaming || extracting}
            extracting={extracting}
            isStreaming={isStreaming}
            onCancel={onCancel}
            slug={slug}
            context={context}
            attachMenuItems={attachMenuItems}
            leading={composerLeading}
            hint={composerNotice ? () => composerNotice : undefined}
          />
        </ComposerDock>
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
