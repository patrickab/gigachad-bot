"use client"

import { useCallback, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import type { Message, Attachment } from "@/lib/domain/types"
import { ChatInput, type AttachMenuItem, type ChatContext, type ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { AttachmentPreview } from "@/features/chat/preview/AttachmentPreview"
import { GitFork } from "lucide-react"
import { ConversationColumn, scrollToTurn, type ConversationReveal, type ConversationTurn } from "@/features/conversation/ConversationColumn"
import { currentRange, scrollToRange, useTextHighlights, type TextHighlight, type TextMatcher } from "@/features/conversation/useTextHighlights"
import { TurnEditsPaused } from "@/features/conversation/TurnQuestion"
import { ChatPairBody, chatPairPreview, pairLabel } from "./ChatPair"
import { findAttachment } from "@/features/chat/attachments"
import { keyPairs, pairMessages, type Pair } from "@/lib/domain/pairs"
import { ComposerDock, useComposerDock } from "./ComposerDock"

/** A search hit in a pair: the `ordinal`-th match of `query` in its question or answer text, as the chat search counts it. */
export interface ChatRevealHit {
  query: string
  role: "user" | "assistant"
  ordinal: number
}

/**
 * Asks the chat of `historyFile` to open and scroll to one question, the column's `reveal` by pair; a new `seq` re-fires it for the same index.
 * With a `hit` it scrolls to the match and paints the query's matches until a reveal without one, or another chat.
 */
export interface ChatReveal {
  historyFile: string | null
  qaIndex: number | null
  hit?: ChatRevealHit | null
  seq: number
}

/** The chat search's matching: the query verbatim, case-insensitive (`ChatStore.search`). */
const chatMatcher: TextMatcher = (query) => {
  if (!query) return null
  const pattern = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi")
  return (text) => [...text.matchAll(pattern)].map((match) => ({ start: match.index, end: match.index + match[0].length }))
}

/** The painted query of a revealed hit in the pair whose question is at message `globalIndex`. */
function chatHighlight(hit: ChatRevealHit, globalIndex: number): TextHighlight {
  return { name: "chat-search", query: hit.query, matcher: chatMatcher, attribute: "data-chat-text", current: { value: `${globalIndex}:${hit.role}`, ordinal: hit.ordinal } }
}

/** What the chat does with its attachments: composer chips and the previews of message attachments. */
export interface AttachmentActions {
  onToggleActive: (messageIndex: number, attachmentName: string) => void
  onRemove: (messageIndex: number, attachmentName: string) => void
  onContentChange: (messageIndex: number, attachmentName: string, newContent: string) => void | Promise<void>
  /** Re-read the reference `path`, so its preview shows the file as it is now. */
  onPreviewOpen: (path: string) => Promise<void>
}

export interface ChatContainerProps {
  chatId: string
  /** Conversation the composer keeps its unsent text for, see `ChatInput`. */
  draftKey?: string | null
  /** The saved history shown, null for an unsaved chat: only a reveal for it applies here. */
  historyFile?: string | null
  messages: Message[]
  /** The chat's saved history is still loading: its empty `messages` must not show the empty state. */
  historyLoading?: boolean
  isStreaming: boolean
  onSend: (text: string, attachments: Attachment[]) => void
  onCancel: () => void
  /** Delete the pair whose question is at message `index`, confirmed by the user. */
  onDeletePair: (index: number) => void
  /** Replace the pair whose question is at message `index`, and every later one, by the edited `text`. Unset -> no edit. */
  onEditPair?: (index: number, text: string) => void
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

/** `entries`, or the last list while it holds the same `(messageIndex, attachment)` pairs: a streamed frame skips the composer. */
function useKeptWhileSame(entries: ChatContext["attachments"]): ChatContext["attachments"] {
  const [kept, setKept] = useState(entries)
  if (entries.length === kept.length && entries.every((entry, i) => entry.messageIndex === kept[i].messageIndex && entry.attachment === kept[i].attachment)) return kept
  setKept(entries)
  return entries
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
  historyFile = null,
  messages,
  historyLoading = false,
  isStreaming,
  onSend,
  onCancel,
  onDeletePair,
  onEditPair,
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
  // Held here, not in the column that remounts per chat: wide mode stays when another chat opens
  const [wide, setWide] = useState(false)

  const pairs = useMemo(() => pairMessages(messages), [messages])
  // Read when an edit is submitted: a count handed to each pair would change for every settled pair on each send and delete
  const pairsRef = useRef(pairs)
  useLayoutEffect(() => { pairsRef.current = pairs }, [pairs])
  const laterTurnsOf = useCallback((index: number) => pairsRef.current.filter((pair) => pair.globalIndex > index).length, [])
  const keys = usePairKeys(pairs)
  const turns = useMemo(() => pairs.map((pair, index): ChatTurn => ({
    key: keys[index],
    label: pairLabel(pair.user),
    live: isStreaming && index === pairs.length - 1,
    pair,
  })), [pairs, keys, isStreaming])
  // A reveal is settled once, when its own chat has loaded: there its pair's key, or nothing when the pair does not exist.
  // Another chat's reveal never applies here, and a settled one does not wait for later messages
  const [settled, setSettled] = useState<{ seq: number | null; historyFile: string | null; reveal: ConversationReveal | null; highlight: TextHighlight | null }>({ seq: null, historyFile, reveal: null, highlight: null })
  if (settled.historyFile !== historyFile) {
    setSettled({ seq: settled.seq, historyFile, reveal: null, highlight: null })
  } else if (reveal && historyFile !== null && reveal.historyFile === historyFile && reveal.seq !== settled.seq && !historyLoading) {
    const turn = reveal.qaIndex === null ? undefined : turns[reveal.qaIndex]
    const highlight = turn && reveal.hit ? chatHighlight(reveal.hit, turn.pair.globalIndex) : null
    const key = turn?.key
    setSettled({
      seq: reveal.seq,
      historyFile,
      highlight,
      reveal: key === undefined ? null : {
        key,
        seq: reveal.seq,
        scroll: highlight ? (root) => {
          const range = currentRange(root, highlight)
          if (!range) return scrollToTurn(root, key)
          scrollToRange(root, range)
          return true
        } : undefined,
      },
    })
  }
  const current = settled.historyFile === historyFile ? settled : null
  const columnReveal = current?.reveal ?? null
  const scrollRef = useRef<HTMLDivElement>(null)
  useTextHighlights(scrollRef, current?.highlight ?? null)

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

  // An open preview of a removed message, or of one whose index moves, closes with it
  const handleDelete = useCallback((index: number) => {
    setPreviewTarget((target) => (target && target.messageIndex >= index ? null : target))
    onDeletePair(index)
  }, [onDeletePair])
  const handleEdit = useMemo(() => onEditPair && ((index: number, text: string) => {
    setPreviewTarget((target) => (target && target.messageIndex >= index ? null : target))
    setFollowSeq((seq) => seq + 1)
    onEditPair(index, text)
  }), [onEditPair])

  const attachments = useKeptWhileSame(messages.flatMap((m, messageIndex) => (m.attachments ?? []).map((attachment) => ({ messageIndex, attachment }))))
  const context = useMemo<ChatContext | undefined>(
    () => attachmentActions && {
      attachments,
      onToggleActive: attachmentActions.onToggleActive,
      onRemove: attachmentActions.onRemove,
      onOpen: (messageIndex, attachment) => void openPreview(messageIndex, attachment),
    },
    [attachmentActions, attachments, openPreview],
  )

  // Settled pairs get equal props each streamed frame, send and delete of another pair, so the column skips them
  const renderPreview = (turn: ChatTurn) => chatPairPreview(turn.pair.assistant)
  const previewText = (turn: ChatTurn) => turn.pair.assistant.content
  const renderBody = (turn: ChatTurn, _index: number, collapse: (() => void) | undefined) => (
    <ChatPairBody
      user={turn.pair.user}
      assistant={turn.pair.assistant}
      globalIndex={turn.pair.globalIndex}
      qaIndex={turn.pair.qaIndex}
      isStreaming={turn.live}
      onCollapse={collapse}
      onAttachmentClick={openPreview}
      onDelete={handleDelete}
      onEdit={handleEdit}
      laterTurnsOf={laterTurnsOf}
      onBranch={onBranch}
    />
  )
  const renderAfter = (turn: ChatTurn) => (branchMessageIdx != null && turn.pair.qaIndex === branchMessageIdx ? <BranchSeparator at={branchMessageIdx} /> : null)

  return (
    <div className="absolute inset-0 flex">
      <div ref={mountColumn} className="flex-1 min-w-0 flex flex-col relative">
        {/* Another chat opens at its end again. No pair is edited or deleted while a reply streams */}
        <TurnEditsPaused.Provider value={isStreaming}>
          <ConversationColumn
            key={chatId}
            turns={turns}
            renderPreview={renderPreview}
            previewText={previewText}
            renderBody={renderBody}
            renderAfter={renderAfter}
            reveal={columnReveal}
            scrollRef={scrollRef}
            wide={wide}
            onWideChange={setWide}
            followSeq={followSeq}
            paddingBottom={dock.paddingBottom}
            onColumnChange={onChatColumnChange}
          />
        </TurnEditsPaused.Provider>
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
            hint={composerNotice}
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
