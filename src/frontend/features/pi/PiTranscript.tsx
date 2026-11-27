"use client"

import { HeldMessages } from "./HeldMessages"
import type { HeldMessage } from "./outbox"
import { createContext, Fragment, memo, useCallback, useContext, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState, type ReactNode, type Ref } from "react"
import { Brain, ChevronRight, FoldVertical, Sparkles, TerminalSquare, type LucideIcon } from "lucide-react"
import { PREVIEW_HEAD_CHARS, previewHead, TurnPreview, TurnSurface, usePreviewInteraction } from "@gigachad/chat-ui"
import "@gigachad/chat-ui/turn.css"
import { LaTeXMarkdown, type FileLinks } from "@/components/content/LaTeXMarkdown"
import { useEventListener } from "@/hooks/useEventListener"
import { TabContext, useTabActive } from "@/contexts/TabContext"
import { MessageAttachments } from "@/features/chat/messages/MessageAttachments"
import { AttachmentPreview } from "@/features/chat/preview/AttachmentPreview"
import { ConversationColumn, type ConversationReveal, type ConversationTurn } from "@/features/conversation/ConversationColumn"
import { TurnAnswer, TurnEditsPaused, TurnQuestion } from "@/features/conversation/TurnQuestion"
import { scrollToRange } from "@/features/conversation/useTextHighlights"
import type { Attachment } from "@/lib/domain/types"
import { resolveLocalFileHref } from "@/lib/fileLinks"
import { isRecord } from "@/lib/utils"
import { factoryRulings } from "./factoryCard"
import { createNoticeKeys, FactoryViewProvider, useTabAgentText, type FactoryView } from "./factoryView"
import { PiSearchBar } from "./PiSearchBar"
import { clearPiSearchHit, stepPiSearchHit, type PiSearchFocus } from "./searchFocus"
import { currentRange, ENTRY_TEXT, useSearchHighlights } from "./useSearchHighlights"
import { renderCustomMessage, renderToolCall, toolPlacement } from "./renderers"
import { ProcessDetails } from "@/features/conversation/tools/ProcessDetails"
import { ToolCallRow } from "@/features/conversation/tools/ToolCallRow"
import { toolImageAttachments, toolImagesOf, userImageAttachments } from "./toolImages"
import { committedRunsOf, shownMessages, splitSkill, textOf, turnsOf, withLiveRuns, type MessageMarks, type SkillBlock, type Transcript, type Turn } from "./transcript"
import type { AgentMessage, AssistantMessage, BashExecutionMessage, CompactionSummaryMessage, CustomMessage, ImageContent, ToolCall, ToolRun } from "./types"
import "./FactoryNotices"
import "./FactoryTool"
import "./MemoryTool"
import { isSilentShellNotice } from "./ShellNotices"
// Registers the `video-preview` card (`~/.pi/agent/extensions/video-paste.ts`, from `~/.os`)
import "./SubagentMessages"
import "./VideoPreview"

const MAX_LABEL_LENGTH = 80

/** One rendered part of an answer, in the order Pi produced it. */
type Piece =
  /** `entryId`: the persisted entry the text belongs to, what a search hit addresses. */
  | { kind: "text"; key: string; text: string; entryId?: string }
  | { kind: "thinking"; key: string; text: string; messageKey: string }
  | { kind: "tool"; key: string; call: ToolCall; messageKey: string }
  /** The images a tool returned, shown as image attachments right after its call, never inside the fold. */
  | { kind: "images"; key: string; attachments: Attachment[] }
  | { kind: "bash"; key: string; message: BashExecutionMessage }
  | { kind: "custom"; key: string; message: CustomMessage }
  /** `retried`: the error of a failed attempt Pi's auto-retry omitted from context, shown instead of the attempt. */
  | { kind: "mark"; key: string; marks: MessageMarks; retried?: string }
  | { kind: "error"; key: string; text: string }

/** Calls of these tools are always visible between the texts, never folded (a `read` shows what the agent looked at). */
const ALWAYS_VISIBLE_TOOLS = new Set(["read"])

/** Folds into "Process details" when settled; a tool with a `standalone` renderer, and a `read`, stay between the texts. */
function isActivity(piece: Piece): boolean {
  return piece.kind === "thinking" || piece.kind === "bash" || (piece.kind === "tool" && toolPlacement(piece.call.name) === "inline" && !ALWAYS_VISIBLE_TOOLS.has(piece.call.name))
}

function assistantPieces(message: AssistantMessage, messageKey: string, runs: Record<string, ToolRun>): Piece[] {
  const pieces = message.content.flatMap((block, index): Piece[] => {
    const key = `${messageKey}-${index}`
    if (block.type === "text") return block.text.trim() ? [{ kind: "text", key, text: block.text }] : []
    if (block.type === "thinking") return block.thinking.trim() ? [{ kind: "thinking", key, text: block.thinking, messageKey }] : []
    const attachments = toolImageAttachments(block.name, toolImagesOf(runs[block.id]))
    return [{ kind: "tool", key, call: block, messageKey }, ...(attachments.length > 0 ? [{ kind: "images" as const, key: `${key}-images`, attachments }] : [])]
  })
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    pieces.push({ kind: "error", key: `${messageKey}-error`, text: message.errorMessage || (message.stopReason === "aborted" ? "Aborted" : "Error") })
  }
  return pieces
}

/** Confirmed state the turns read their entry edits and branch points from. */
interface Confirmed {
  count: number
  entryIds: string[]
  marks: Record<string, MessageMarks>
}

function marksAt(confirmed: Confirmed, index: number): MessageMarks | undefined {
  const marks = index < confirmed.count ? confirmed.marks[confirmed.entryIds[index]] : undefined
  return marks && (marks.context || marks.label) ? marks : undefined
}

/** The error of a failed attempt Pi's auto-retry left in history and omitted from context, else undefined. */
function retriedError(message: AgentMessage, marks: MessageMarks | undefined): string | undefined {
  if (marks?.context !== "omitted" || message.role !== "assistant") return undefined
  const { stopReason, errorMessage } = message as AssistantMessage
  return stopReason === "error" ? (errorMessage || "Error").replace(/\.$/, "") : undefined
}

/** The answer's pieces: committed messages after the prompt (each edited one after its marks), then the streaming reply when live. A retried attempt is only its mark. */
function answerPieces(messages: AgentMessage[], turn: Turn, streaming: AssistantMessage | null, confirmed: Confirmed, runs: Record<string, ToolRun>): Piece[] {
  const pieces: Piece[] = []
  for (let index = turn.start + 1; index < turn.end; index++) {
    const message = messages[index]
    const key = `m${index}`
    const visible = message.role === "assistant" || message.role === "bashExecution" || (message.role === "custom" && (message as CustomMessage).display && !isSilentShellNotice(message))
    const marks = visible ? marksAt(confirmed, index) : undefined
    const retried = retriedError(message, marks)
    if (marks) pieces.push({ kind: "mark", key: `${key}-mark`, marks, retried })
    if (retried) continue
    if (message.role === "assistant") {
      const entryId = index < confirmed.count ? confirmed.entryIds[index] : undefined
      pieces.push(...assistantPieces(message as AssistantMessage, key, runs).map((piece) => (piece.kind === "text" && entryId ? { ...piece, entryId } : piece)))
    }
    if (message.role === "bashExecution") pieces.push({ kind: "bash", key, message: message as BashExecutionMessage })
    if (message.role === "custom" && (message as CustomMessage).display && !isSilentShellNotice(message)) pieces.push({ kind: "custom", key, message: message as CustomMessage })
  }
  if (streaming) pieces.push(...assistantPieces(streaming, "streaming", runs))
  return pieces
}

/** A folded block: its icon and label, `children` on expand. */
function Fold({ icon: Icon, label, children }: { icon: LucideIcon; label: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="my-1 text-xs text-ink-muted">
      <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)} className="flex items-center gap-1.5 rounded px-1 py-0.5 hover:text-ink focus-visible:outline focus-visible:outline-1">
        <Icon size={12} aria-hidden="true" />
        <ChevronRight size={12} aria-hidden="true" className={open ? "rotate-90 transition-transform" : "transition-transform"} />
        {label}
      </button>
      {open && children}
    </div>
  )
}

/** Memoized like the tool rows: a live turn re-renders on each streamed frame, its settled pieces must not. */
const ThinkingBlock = memo(function ThinkingBlock({ text }: { text: string }) {
  return <Fold icon={Brain} label="Thinking"><p className="mt-1 whitespace-pre-wrap pl-5 leading-5">{text}</p></Fold>
})

const PreviewContext = createContext<(attachment: Attachment) => void>(() => {})

/** A tool's images as the chat's image attachments, a click opens the image in a floating window. */
function ToolImages({ attachments }: { attachments: Attachment[] }) {
  const open = useContext(PreviewContext)
  return <MessageAttachments attachments={attachments} onClick={open} />
}

const BashBlock = memo(function BashBlock({ message }: { message: BashExecutionMessage }) {
  return (
    <div className="my-1 font-mono text-xs">
      <div className="flex items-center gap-1.5 text-ink-muted"><TerminalSquare size={12} aria-hidden="true" />$ {message.command}</div>
      {message.output && <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap text-ink-subtle">{message.output}</pre>}
    </div>
  )
})

/** Quiet line naming a message's label and `context_edit`, or the error a retried attempt failed with. */
function MarksLine({ marks, retried }: { marks: MessageMarks; retried?: string }) {
  const context = retried ? `Retried after: ${retried}` : marks.context === "omitted" ? "Omitted from context" : marks.context === "replaced" ? "Replaced in context" : undefined
  const parts = [marks.label, context].filter(Boolean)
  return <p className="my-1 text-xs text-ink-faint">{parts.join(" · ")}</p>
}

type NoticeJump = (key: string) => void

const fileOpeners = new WeakMap<FileLinks, (path: string, line?: number) => void>()

/**
 * One opener per `fileLinks`: a tool row is memoized on it, a new function per render would re-render every row of a live turn on each streamed frame.
 * A `line` (a finding's `file:line`) goes to `fileLinks.open`, which opens the file there where it can.
 */
function fileOpener(fileLinks: FileLinks): (path: string, line?: number) => void {
  let open = fileOpeners.get(fileLinks)
  if (!open) {
    open = (path, line) => fileLinks.open(resolveLocalFileHref(path, fileLinks.base) ?? path, line)
    fileOpeners.set(fileLinks, open)
  }
  return open
}

function renderPiece(piece: Piece, runs: Record<string, ToolRun>, live: boolean, fileLinks: FileLinks): ReactNode {
  switch (piece.kind) {
    case "text": {
      const markdown = <LaTeXMarkdown content={piece.text} streaming={live && piece.key.startsWith("streaming")} fileLinks={fileLinks} />
      return piece.entryId ? <div key={piece.key} data-entry-id={piece.entryId} className="contents">{markdown}</div> : <Fragment key={piece.key}>{markdown}</Fragment>
    }
    case "thinking": return <ThinkingBlock key={piece.key} text={piece.text} />
    case "tool": {
      const run = runs[piece.call.id]
      const row = () => <ToolCallRow call={piece.call} run={run} streaming={live} onOpenFile={fileOpener(fileLinks)} />
      return <Fragment key={piece.key}>{renderToolCall(piece.call, run, row)}</Fragment>
    }
    case "images": return <ToolImages key={piece.key} attachments={piece.attachments} />
    case "bash": return <BashBlock key={piece.key} message={piece.message} />
    case "custom": {
      const text = () => <LaTeXMarkdown content={textOf(piece.message.content)} fileLinks={fileLinks} />
      return <Fragment key={piece.key}>{renderCustomMessage(piece.message, text)}</Fragment>
    }
    case "mark": return <MarksLine key={piece.key} marks={piece.marks} retried={piece.retried} />
    case "error": return <p key={piece.key} className="my-1 text-xs text-danger">{piece.text}</p>
  }
}

/** Newest pieces of a live turn's last run of activity that stay unfolded, so tool progress is in view. */
const LIVE_FLAT_PIECES = 12

/**
 * The answer like omp-web shows it: text stays visible, every run of activity (thinking, tool calls, shell commands) between
 * texts folds into "Process details". `read` calls and the images tools returned (image attachments) are never folded.
 * A live turn keeps the newest `LIVE_FLAT_PIECES` of its last run unfolded so tool progress is in view: a long agent turn then
 * mounts a handful of rows, not hundreds. The run folds whole once the turn settles.
 */
function Answer({ pieces, runs, live, fileLinks }: { pieces: Piece[]; runs: Record<string, ToolRun>; live: boolean; fileLinks: FileLinks }) {
  const rendered: ReactNode[] = []
  let activity: Piece[] = []
  const flush = (last: boolean) => {
    if (activity.length === 0) return
    const flat = live && last ? activity.slice(-LIVE_FLAT_PIECES) : []
    const group = activity.slice(0, activity.length - flat.length)
    if (group.length > 0) {
      rendered.push(
        <ProcessDetails
          key={`activity-${group[0].key}`}
          messageCount={new Set(group.flatMap((piece) => ("messageKey" in piece ? [piece.messageKey] : [piece.key]))).size}
          toolCallCount={group.filter((piece) => piece.kind === "tool").length}
        >
          {() => group.map((piece) => renderPiece(piece, runs, live, fileLinks))}
        </ProcessDetails>,
      )
    }
    for (const piece of flat) rendered.push(renderPiece(piece, runs, live, fileLinks))
    activity = []
  }
  for (const piece of pieces) {
    if (isActivity(piece)) {
      activity.push(piece)
      continue
    }
    flush(false)
    rendered.push(renderPiece(piece, runs, live, fileLinks))
  }
  flush(true)
  return <>{rendered}</>
}

/** The pieces up to the text that fills a collapsed turn's preview, that text cut to its head. Runs of activity stay whole, so their counts hold. */
function headPieces(pieces: Piece[]): Piece[] {
  let budget = PREVIEW_HEAD_CHARS
  const head: Piece[] = []
  for (const piece of pieces) {
    if (piece.kind !== "text") {
      head.push(piece)
      continue
    }
    const text = previewHead(piece.text, budget)
    head.push(text === piece.text ? piece : { ...piece, text })
    budget -= text.length
    if (budget <= 0 || text !== piece.text) break
  }
  return head
}

/** A collapsed turn's answer: its head, the whole once the reader scrolls or focuses the preview. A long agent turn then holds a fraction of its elements while collapsed. */
function CollapsedAnswer({ pieces, runs, fileLinks }: { pieces: Piece[]; runs: Record<string, ToolRun>; fileLinks: FileLinks }) {
  const { full, attach } = usePreviewInteraction()
  return <div ref={attach} className="contents"><Answer pieces={full ? pieces : headPieces(pieces)} runs={runs} live={false} fileLinks={fileLinks} /></div>
}

function clip(text: string): string {
  const firstLine = text.trim().split("\n")[0] ?? ""
  return firstLine.length <= MAX_LABEL_LENGTH ? firstLine : `${firstLine.slice(0, MAX_LABEL_LENGTH)}\u2026`
}

function promptOf(message: AgentMessage): { text: string; compaction: boolean; skill?: SkillBlock } {
  if (message.role === "compactionSummary") return { text: (message as CompactionSummaryMessage).summary, compaction: true }
  return { ...splitSkill(textOf((message as { content: string | { type: string; text?: string }[] }).content)), compaction: false }
}

/** The prompt's own image blocks (base64), as sent: what an edit sends again. */
function promptImageBlocks(message: AgentMessage): ImageContent[] {
  const { content } = message as { content?: unknown }
  if (!Array.isArray(content)) return []
  return content.filter((block): block is ImageContent => block?.type === "image" && typeof block.data === "string")
}

/** The skill a prompt loaded, folded like "Thinking": its name, the skill file on expand. */
function SkillFold({ skill, fileLinks }: { skill: SkillBlock; fileLinks: FileLinks }) {
  return <Fold icon={Sparkles} label={<>Skill · {skill.name}</>}><div className="mt-1 pl-5"><LaTeXMarkdown content={skill.body} fileLinks={fileLinks} /></div></Fold>
}

/**
 * What one turn renders from: its own messages (`turn` spans them all), their entry ids and marks, and the runs of its tool
 * calls. Kept while equal (`useTurnViews`), so a send, a delete or a tool run elsewhere hands a settled turn the same view.
 */
interface TurnView {
  messages: AgentMessage[]
  turn: Turn
  confirmed: Confirmed
  runs: Record<string, ToolRun>
}

function turnView(messages: AgentMessage[], turn: Turn, confirmed: Confirmed, runs: Record<string, ToolRun>, live: boolean): TurnView {
  const own = messages.slice(turn.start, turn.end)
  const entryIds = confirmed.entryIds.slice(turn.start, Math.min(turn.end, confirmed.count))
  const marks: Record<string, MessageMarks> = {}
  for (const id of entryIds) if (confirmed.marks[id]) marks[id] = confirmed.marks[id]
  // A live turn's streaming reply has calls no message holds yet: it reads every run
  const ownRuns: Record<string, ToolRun> = live ? runs : {}
  if (!live) {
    for (const message of own) {
      if (message.role !== "assistant") continue
      for (const block of (message as AssistantMessage).content) if (block.type === "toolCall" && runs[block.id]) ownRuns[block.id] = runs[block.id]
    }
  }
  return { messages: own, turn: { start: 0, end: own.length }, confirmed: { count: entryIds.length, entryIds, marks }, runs: ownRuns }
}

function sameItems<T>(a: readonly T[], b: readonly T[]): boolean {
  return a.length === b.length && a.every((item, index) => Object.is(item, b[index]))
}

function sameRecord<T>(a: Record<string, T>, b: Record<string, T>): boolean {
  const keys = Object.keys(a)
  return keys.length === Object.keys(b).length && keys.every((key) => Object.is(a[key], b[key]))
}

function sameView(a: TurnView, b: TurnView): boolean {
  return sameItems(a.messages, b.messages) && sameItems(a.confirmed.entryIds, b.confirmed.entryIds)
    && sameRecord(a.confirmed.marks, b.confirmed.marks) && sameRecord(a.runs, b.runs)
}

/**
 * Each turn's view by key, the previous object while its content is equal.
 *
 * With the same `turns` object (no message, entry, mark or live-flag change) only `runs` changed, and live tool events belong
 * to the running turn: a new turn starts with a user message, which Pi delivers only after the running tools ended. So every
 * other turn keeps its stored view and a tool-output frame reads the live turn only.
 */
function useTurnViews(turns: readonly PiColumnTurn[], messages: AgentMessage[], confirmed: Confirmed, runs: Record<string, ToolRun>): Map<string, TurnView> {
  const [state, setState] = useState<{ turns: readonly PiColumnTurn[]; runs: Record<string, ToolRun>; views: Map<string, TurnView> }>({ turns: [], runs: {}, views: new Map() })
  if (state.turns === turns && state.runs === runs) return state.views
  const sameTurns = state.turns === turns
  const views = new Map<string, TurnView>()
  for (const item of turns) {
    const kept = state.views.get(item.key)
    if (sameTurns && kept && !item.live) {
      views.set(item.key, kept)
      continue
    }
    const fresh = turnView(messages, item.turn, confirmed, runs, item.live)
    views.set(item.key, kept && sameView(kept, fresh) ? kept : fresh)
  }
  setState({ turns, runs, views })
  return views
}

interface PiTurnProps {
  view: TurnView
  fileLinks: FileLinks
}

/** A collapsed turn's preview: the answer with activity folded as when opened (omp-web's preview); memoized like the open turn. */
const PiTurnPreview = memo(function PiTurnPreview({ view, fileLinks }: PiTurnProps) {
  const { messages, turn, confirmed, runs } = view
  const pieces = answerPieces(messages, turn, null, confirmed, runs)
  return <TurnPreview tools={[]}>{pieces.length > 0 ? <CollapsedAnswer pieces={pieces} runs={runs} fileLinks={fileLinks} /> : null}</TurnPreview>
})

/** One open turn's question and answer; memoized so a streamed delta re-renders the newest turn only. */
const PiTurnBody = memo(function PiTurnBody({ view, turnKey, streaming, live, laterTurnsOf, onCollapse, onBranch, onDeleteTurn, onEditTurn, fileLinks }: PiTurnProps & {
  turnKey: string
  /** The streaming reply, for the newest turn only. */
  streaming: AssistantMessage | null
  live: boolean
  /** Counts the turns after the one keyed `key`, what an edit removes with it. */
  laterTurnsOf: (key: string) => number
  onCollapse?: () => void
  /** Branch keeping `entryId`. */
  onBranch?: (entryId: string) => void
  /** Delete the turn whose prompt is `entryId`. */
  onDeleteTurn?: (entryId: string) => void
  /** Delete the turn whose prompt is `entryId` and every later one, then send `text` with `images`. */
  onEditTurn?: (entryId: string, text: string, images: ImageContent[]) => void
}) {
  const { messages, turn, confirmed, runs } = view
  const prompt = messages[turn.start]
  const { text, compaction, skill } = promptOf(prompt)
  const promptImages = useMemo(() => userImageAttachments((prompt as { content?: unknown }).content), [prompt])
  const pieces = answerPieces(messages, turn, streaming, confirmed, runs)
  const promptMarks = marksAt(confirmed, turn.start)
  const promptEntryId = turn.start < confirmed.count ? confirmed.entryIds[turn.start] : undefined
  const icon = compaction ? <FoldVertical size={14} aria-hidden="true" /> : undefined
  const answerText = pieces.flatMap((piece) => (piece.kind === "text" ? [piece.text] : [])).join("\n\n")
  // Branch from the turn's last persisted entry; a turn with none yet (live, optimistic) has no branch point
  const lastConfirmed = Math.min(turn.end, confirmed.count) - 1
  const branchFrom = !live && onBranch && lastConfirmed >= turn.start ? confirmed.entryIds[lastConfirmed] : undefined
  // Edit and delete address the prompt's entry: a turn without one yet, and a compaction, have neither
  const editFrom = !live && !compaction ? promptEntryId : undefined
  // A skill prompt is edited as its args, sent again as the command that loads the skill
  const edit = editFrom && onEditTurn ? (draft: string) => onEditTurn(editFrom, skill ? `/skill:${skill.name} ${draft}` : draft, promptImageBlocks(prompt)) : undefined
  const remove = editFrom && onDeleteTurn ? () => onDeleteTurn(editFrom) : undefined
  return (
    <>
      <TurnQuestion text={text} live={live} icon={icon} onCollapse={onCollapse} onEdit={edit} onDelete={remove} laterTurns={() => laterTurnsOf(turnKey)}>
        {promptMarks && <MarksLine marks={promptMarks} />}
        {skill && <SkillFold skill={skill} fileLinks={fileLinks} />}
        {text || !skill ? <div data-entry-id={promptEntryId}><LaTeXMarkdown content={text} fileLinks={fileLinks} /></div> : null}
        {promptImages.length > 0 && <ToolImages attachments={promptImages} />}
      </TurnQuestion>
      {pieces.length > 0 && (
        <TurnSurface>
          <TurnAnswer
            text={answerText}
            live={live}
            onBranch={branchFrom ? () => onBranch!(branchFrom) : undefined}
            branchLabel="Branch session from this point"
            tail={isActivity(pieces[pieces.length - 1]) ? "activity" : "text"}
          >
            <Answer pieces={pieces} runs={runs} live={live} fileLinks={fileLinks} />
          </TurnAnswer>
        </TurnSurface>
      )}
    </>
  )
})

/** A turn in the column: its messages' range. */
interface PiColumnTurn extends ConversationTurn {
  turn: Turn
}

/**
 * Every turn as the regular chat's Q&A card in the shared `ConversationColumn` (paging, following the end, open state, width,
 * wide mode): the question and the answer with the shared turn actions (`TurnQuestion`, `TurnAnswer`), activity folded into
 * "Process details". File links open through `fileLinks`.
 *
 * - A turn's key is its prompt's entry id, a pending prompt's its position until its entry arrives
 * - A search hit (`search`) reveals its turn, scrolls to the hit, centred, and holds the view there; Enter and Shift+Enter
 *   (outside text fields) step through the session's hits, Escape clears
 * - `onBranch` adds "Branch session from this point" to settled turns, with the entry id of the turn's last persisted message
 * - `onDeleteTurn` and `onEditTurn` add delete and edit to the question of each turn with a persisted prompt, compactions
 *   aside, and to none while a run is live (`TurnEditsPaused`). Both get the prompt's entry id; the transcript follows the host's resync
 * - A settled turn's props hold through streamed deltas, sends, deletes and tool runs of other turns (`useTurnViews`)
 * - `onColumnChange` gets the turn column, for the top bar's model name to make way for it like a chat's
 * - `readOnly` (a session preview): keys never reach its column
 */
export function PiTranscript({ transcript, held = [], onRetryHeld, onDeleteHeld, fileLinks, search = null, onBranch, onDeleteTurn, onEditTurn, paddingBottom, onColumnChange, readOnly, revealNoticeRef }: {
  transcript: Transcript
  /** Sent but unconfirmed messages, shown below the turns. */
  held?: readonly HeldMessage[]
  onRetryHeld?: (id: string) => void
  onDeleteHeld?: (id: string) => void
  fileLinks: FileLinks
  /** The session search hit to show: its turn is mounted and opened, the view scrolls to it and every hit is painted (CSS Custom Highlight API). */
  search?: PiSearchFocus | null
  onBranch?: (entryId: string) => void
  onDeleteTurn?: (entryId: string) => void
  onEditTurn?: (entryId: string, text: string, images: ImageContent[]) => void
  paddingBottom?: number
  onColumnChange?: (column: HTMLDivElement | null) => void
  readOnly?: boolean
  /** Receives the jump to a factory notice's card (by its key), for the tab's waves strip: it reveals the notice's turn, mounted or not. */
  revealNoticeRef?: Ref<NoticeJump | null>
}) {
  const { messages: confirmedMessages, unconfirmed, optimistic, entryIds, marks, streaming, running, runs: liveRuns } = transcript
  // Derived from the fields that change per message, not per streamed delta, so settled turns keep their props
  const messages = useMemo(() => shownMessages({ messages: confirmedMessages, unconfirmed, optimistic }), [confirmedMessages, unconfirmed, optimistic])
  const turns = useMemo(() => turnsOf(messages), [messages])
  // Committed runs walk the messages once per message; a tool-output frame only lays the live runs over them
  const committedRuns = useMemo(() => committedRunsOf(messages), [messages])
  const runs = useMemo(() => withLiveRuns(committedRuns, liveRuns), [committedRuns, liveRuns])
  const confirmed = useMemo<Confirmed>(() => ({ count: confirmedMessages.length, entryIds, marks }), [confirmedMessages.length, entryIds, marks])
  const liveTail = running || streaming !== null || optimistic !== null
  const deleteTurn = readOnly ? undefined : onDeleteTurn
  const editTurn = readOnly ? undefined : onEditTurn
  const columnTurns = useMemo(() => turns.map((turn, index): PiColumnTurn => {
    const { text, compaction, skill } = promptOf(messages[turn.start])
    return {
      key: (turn.start < confirmed.count ? confirmed.entryIds[turn.start] : undefined) ?? `pending-${turn.start}`,
      label: compaction ? "Compaction" : clip(text) || (skill ? `/skill:${skill.name}` : "(empty)"),
      icon: compaction ? <FoldVertical size={14} aria-hidden="true" /> : undefined,
      live: index === turns.length - 1 && liveTail,
      turn,
    }
  }), [turns, messages, confirmed, liveTail])
  const views = useTurnViews(columnTurns, messages, confirmed, runs)
  // Read when an edit is submitted: a count handed to each turn would change for every settled turn on each send and delete
  const keysRef = useRef<string[]>([])
  useLayoutEffect(() => { keysRef.current = columnTurns.map((item) => item.key) }, [columnTurns])
  const laterTurnsOf = useCallback((key: string) => {
    const index = keysRef.current.indexOf(key)
    return index < 0 ? 0 : keysRef.current.length - 1 - index
  }, [])
  const [preview, setPreview] = useState<Attachment | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // The jump to a search hit: the column mounts and opens its turn, then scrolls to its text
  const tabActive = useTabActive()
  useSearchHighlights(scrollRef, search)
  const hit = search?.matches[search.index]
  const hitMessage = hit ? entryIds.indexOf(hit.entryId) : -1
  const hitTurn = hitMessage < 0 || hitMessage >= confirmedMessages.length ? -1 : turns.findIndex((turn) => turn.start <= hitMessage && hitMessage < turn.end)
  const hitKey = hitTurn >= 0 ? columnTurns[hitTurn].key : null
  // Waits for the session's entries: nothing is loaded yet when a pick opens it
  const reveal = useMemo<ConversationReveal | null>(() => (search && hit && hitKey !== null ? {
    key: hitKey,
    seq: search.seq,
    scroll: (root) => {
      const range = currentRange(root, search)
      if (range) {
        scrollToRange(root, range)
        return true
      }
      // Else the entry's text, e.g. when a hit's ordinal is past what rendered (a message too large to render shows less)
      const target = root.querySelector(`${ENTRY_TEXT}[data-entry-id="${CSS.escape(hit.entryId)}"]`)
      if (!target) return false
      target.scrollIntoView({ block: "center", behavior: "instant" })
      return true
    },
  } : null), [search, hit, hitKey])

  // Factory events: the tab's view switch (Alt+V), their files, and a decision's jump to the notice it answers
  const tabId = useContext(TabContext)?.tab.id ?? null
  const agentText = useTabAgentText(tabId, tabActive && !readOnly)
  const navRef = useRef({ messages, turns, columnTurns, search })
  useLayoutEffect(() => { navRef.current = { messages, turns, columnTurns, search } }, [messages, turns, columnTurns, search])
  const jumpSeq = useRef(0)
  const [jump, setJump] = useState<{ reveal: ConversationReveal; search: PiSearchFocus | null } | null>(null)
  const revealNotice = useCallback((noticeKey: string) => {
    const { messages, turns, columnTurns, search } = navRef.current
    const index = messages.findIndex((message) => message.role === "custom" && (message as CustomMessage).customType === "factory-notice"
      && isRecord((message as CustomMessage).details) && ((message as CustomMessage).details as Record<string, unknown>).key === noticeKey)
    const turnIndex = index < 0 ? -1 : turns.findIndex((turn) => turn.start <= index && index < turn.end)
    if (turnIndex < 0) return
    // Negative: a search hit's seq counts up from 1, the column reveals once per seq
    jumpSeq.current -= 1
    setJump({
      search,
      reveal: {
        key: columnTurns[turnIndex].key,
        seq: jumpSeq.current,
        scroll: (root) => {
          const target = root.querySelector(`[data-factory-notice="${CSS.escape(noticeKey)}"]`)
          if (!target) return false
          target.scrollIntoView({ block: "center", behavior: "instant" })
          return true
        },
      },
    })
  }, [])
  useImperativeHandle<NoticeJump | null, NoticeJump | null>(revealNoticeRef, () => (readOnly ? null : revealNotice), [readOnly, revealNotice])
  // The notice keys live in one object: a new notice re-renders only the decisions that answer it
  const [notices] = useState(createNoticeKeys)
  const noticeKeys = useMemo(() => factoryRulings(messages).notices, [messages])
  useLayoutEffect(() => notices.set(noticeKeys), [notices, noticeKeys])
  const factoryView = useMemo<FactoryView>(() => ({ agentText, openFile: fileOpener(fileLinks), revealNotice: readOnly ? undefined : revealNotice, notices }), [agentText, fileLinks, revealNotice, readOnly, notices])
  // The newer of the two: a jump holds until the next search hit
  const shownReveal = jump && jump.search === search ? jump.reveal : reveal
  // Enter and Shift+Enter step, Escape clears: from the page or the transcript, never from a text field or a button
  useEventListener("window", "keydown", (event) => {
    const target = event.target
    if (target instanceof Element && (!(target === document.body || scrollRef.current?.contains(target)) || target.closest("input, textarea, select, button, [contenteditable='true']"))) return
    if (event.key === "Escape") clearPiSearchHit()
    else if (event.key === "Enter") stepPiSearchHit(event.shiftKey ? -1 : 1)
    else return
    event.preventDefault()
  }, { active: tabActive && search !== null })

  const renderPreview = (item: PiColumnTurn) => <PiTurnPreview view={views.get(item.key)!} fileLinks={fileLinks} />
  const renderBody = (item: PiColumnTurn, index: number, collapse: (() => void) | undefined) => (
    <PiTurnBody
      view={views.get(item.key)!}
      turnKey={item.key}
      streaming={index === turns.length - 1 ? streaming : null}
      live={item.live}
      laterTurnsOf={laterTurnsOf}
      onCollapse={collapse}
      onBranch={onBranch}
      onDeleteTurn={deleteTurn}
      onEditTurn={editTurn}
      fileLinks={fileLinks}
    />
  )

  return (
    <PreviewContext.Provider value={setPreview}>
      <FactoryViewProvider value={factoryView}>
        {/* While a run is live no turn is edited or deleted: the host refuses a running session */}
        <TurnEditsPaused.Provider value={liveTail}>
          <ConversationColumn
            turns={columnTurns}
            renderPreview={renderPreview}
            renderBody={renderBody}
            header={search && <PiSearchBar search={search} />}
            footer={<>
              <HeldMessages held={held} onRetry={(id) => onRetryHeld?.(id)} onDelete={(id) => onDeleteHeld?.(id)} onOpenImage={setPreview} />
              <AttachmentPreview attachment={preview} onClose={() => setPreview(null)} />
            </>}
            reveal={shownReveal}
            paddingBottom={paddingBottom}
            onColumnChange={onColumnChange}
            scrollRef={scrollRef}
            readOnly={readOnly}
          />
        </TurnEditsPaused.Provider>
      </FactoryViewProvider>
    </PreviewContext.Provider>
  )
}
