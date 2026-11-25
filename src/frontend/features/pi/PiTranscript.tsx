"use client"

import { HeldMessages } from "./HeldMessages"
import type { HeldMessage } from "./outbox"
import { createContext, Fragment, memo, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { Brain, ChevronRight, CornerUpLeft, FoldVertical, Sparkles, Square, TerminalSquare, Volume2 } from "lucide-react"
import { CopyAction, ForkAction, TurnAction, TurnCard, TurnHeader, TurnPreview, TurnRegion, TurnSurface } from "@gigachad/chat-ui"
import "@gigachad/chat-ui/turn.css"
import { LaTeXMarkdown, type FileLinks } from "@/components/content/LaTeXMarkdown"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { useEventListener } from "@/hooks/useEventListener"
import { useTabActive } from "@/contexts/TabContext"
import { ResizeEdges, useColumnWidth, type ResizeStart } from "@/components/ui/columnResize"
import { MessageAttachments } from "@/features/chat/messages/MessageAttachments"
import { AttachmentPreview } from "@/features/chat/preview/AttachmentPreview"
import { WideModeToggle } from "@/features/chat/WideModeToggle"
import type { Attachment } from "@/lib/domain/types"
import { resolveLocalFileHref } from "@/lib/fileLinks"
import { PiSearchBar } from "./PiSearchBar"
import { clearPiSearchHit, stepPiSearchHit, type PiSearchFocus } from "./searchFocus"
import { currentRange, ENTRY_TEXT, useSearchHighlights } from "./useSearchHighlights"
import { renderCustomMessage, renderToolCall, toolPlacement } from "./renderers"
import { ProcessDetails } from "./tools/ProcessDetails"
import { ToolCallRow } from "./tools/ToolCallRow"
import { toolImageAttachments, toolImagesOf } from "./toolImages"
import { shownMessages, textOf, toolRunsOf, turnsOf, type MessageMarks, type Transcript, type Turn } from "./transcript"
import type { AgentMessage, AssistantMessage, BashExecutionMessage, CompactionSummaryMessage, CustomMessage, ToolCall, ToolRun } from "./types"
import "./ShellNotices"
// Registers the `video-preview` card (`~/.pi/agent/extensions/video-paste.ts`, from `~/.os`)
import "./VideoPreview"

const MAX_LABEL_LENGTH = 80
/** Turns mounted at first and added per "Show earlier turns": older ones create no elements (smooth streaming). */
export const TURN_PAGE_SIZE = 20
/** Distance from the end (px) that still counts as reading the end, which the view then follows. */
const PINNED_SLACK_PX = 48
/** Load the earlier page while the button is still this far below the top edge, so an upward scroll continues into it. */
const LOAD_EARLIER_ROOT_MARGIN = "400px 0px 0px 0px"

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
    const visible = message.role === "assistant" || message.role === "bashExecution" || (message.role === "custom" && (message as CustomMessage).display)
    const marks = visible ? marksAt(confirmed, index) : undefined
    const retried = retriedError(message, marks)
    if (marks) pieces.push({ kind: "mark", key: `${key}-mark`, marks, retried })
    if (retried) continue
    if (message.role === "assistant") {
      const entryId = index < confirmed.count ? confirmed.entryIds[index] : undefined
      pieces.push(...assistantPieces(message as AssistantMessage, key, runs).map((piece) => (piece.kind === "text" && entryId ? { ...piece, entryId } : piece)))
    }
    if (message.role === "bashExecution") pieces.push({ kind: "bash", key, message: message as BashExecutionMessage })
    if (message.role === "custom" && (message as CustomMessage).display) pieces.push({ kind: "custom", key, message: message as CustomMessage })
  }
  if (streaming) pieces.push(...assistantPieces(streaming, "streaming", runs))
  return pieces
}

function ThinkingBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="my-1 text-xs text-ink-muted">
      <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)} className="flex items-center gap-1.5 rounded px-1 py-0.5 hover:text-ink focus-visible:outline focus-visible:outline-1">
        <Brain size={12} aria-hidden="true" />
        <ChevronRight size={12} aria-hidden="true" className={open ? "rotate-90 transition-transform" : "transition-transform"} />
        Thinking
      </button>
      {open && <p className="mt-1 whitespace-pre-wrap pl-5 leading-5">{text}</p>}
    </div>
  )
}

const PreviewContext = createContext<(attachment: Attachment) => void>(() => {})

/** A tool's images as the chat's image attachments, a click opens the lightbox (the file panel). */
function ToolImages({ attachments }: { attachments: Attachment[] }) {
  const open = useContext(PreviewContext)
  return <MessageAttachments attachments={attachments} onClick={open} />
}

function BashBlock({ message }: { message: BashExecutionMessage }) {
  return (
    <div className="my-1 font-mono text-xs">
      <div className="flex items-center gap-1.5 text-ink-muted"><TerminalSquare size={12} aria-hidden="true" />$ {message.command}</div>
      {message.output && <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap text-ink-subtle">{message.output}</pre>}
    </div>
  )
}

/** Quiet line naming a message's label and `context_edit`, or the error a retried attempt failed with. */
function MarksLine({ marks, retried }: { marks: MessageMarks; retried?: string }) {
  const context = retried ? `Retried after: ${retried}` : marks.context === "omitted" ? "Omitted from context" : marks.context === "replaced" ? "Replaced in context" : undefined
  const parts = [marks.label, context].filter(Boolean)
  return <p className="my-1 text-xs text-ink-faint">{parts.join(" · ")}</p>
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
      const row = () => <ToolCallRow call={piece.call} run={run} streaming={live} defaultCollapsed onOpenFile={(path) => fileLinks.open(resolveLocalFileHref(path, fileLinks.base) ?? path)} />
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

/**
 * The answer like omp-web shows it: text stays visible, every run of activity (thinking, tool calls, shell commands) between
 * texts folds into "Process details". `read` calls and the images tools returned (image attachments) are never folded. A live turn stays flat so tool progress is in view, it folds once settled.
 */
function Answer({ pieces, runs, live, fileLinks }: { pieces: Piece[]; runs: Record<string, ToolRun>; live: boolean; fileLinks: FileLinks }) {
  if (live) return <>{pieces.map((piece) => renderPiece(piece, runs, live, fileLinks))}</>
  const rendered: ReactNode[] = []
  let activity: Piece[] = []
  const flush = () => {
    if (activity.length === 0) return
    const group = activity
    rendered.push(
      <ProcessDetails
        key={`activity-${group[0].key}`}
        messageCount={new Set(group.flatMap((piece) => ("messageKey" in piece ? [piece.messageKey] : [piece.key]))).size}
        toolCallCount={group.filter((piece) => piece.kind === "tool").length}
      >
        {() => group.map((piece) => renderPiece(piece, runs, false, fileLinks))}
      </ProcessDetails>,
    )
    activity = []
  }
  for (const piece of pieces) {
    if (isActivity(piece)) {
      activity.push(piece)
      continue
    }
    flush()
    rendered.push(renderPiece(piece, runs, false, fileLinks))
  }
  flush()
  return <>{rendered}</>
}

/** Speaks `text` with the browser's speech synthesis, a second press stops it. A Pi-only answer action, beside the chat's copy and branch. */
function ReadAloudAction({ text }: { text: string }) {
  const [speaking, setSpeaking] = useState(false)
  useEffect(() => () => {
    if (speaking) speechSynthesis.cancel()
  }, [speaking])
  const toggle = () => {
    speechSynthesis.cancel()
    if (speaking) return setSpeaking(false)
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.onend = () => setSpeaking(false)
    utterance.onerror = () => setSpeaking(false)
    speechSynthesis.speak(utterance)
    setSpeaking(true)
  }
  return <TurnAction label={speaking ? "Stop reading" : "Read aloud"} onClick={toggle} icon={speaking ? <Square size={14} aria-hidden="true" /> : <Volume2 size={14} aria-hidden="true" />} />
}

function clip(text: string): string {
  const firstLine = text.trim().split("\n")[0] ?? ""
  return firstLine.length <= MAX_LABEL_LENGTH ? firstLine : `${firstLine.slice(0, MAX_LABEL_LENGTH)}\u2026`
}

interface SkillBlock { name: string; body: string }

/** Pi's expansion of `/skill:<name> <args>`: the skill's file in a `<skill>` block, then the args. */
export function splitSkill(text: string): { skill?: SkillBlock; text: string } {
  const match = /^<skill name="([^"]*)"[^>]*>\n([\s\S]*?)\n<\/skill>(?:\n\n([\s\S]*))?$/.exec(text)
  return match ? { skill: { name: match[1], body: match[2] }, text: match[3] ?? "" } : { text }
}

function promptOf(message: AgentMessage): { text: string; compaction: boolean; skill?: SkillBlock } {
  if (message.role === "compactionSummary") return { text: (message as CompactionSummaryMessage).summary, compaction: true }
  return { ...splitSkill(textOf((message as { content: string | { type: string; text?: string }[] }).content)), compaction: false }
}

/** The skill a prompt loaded, folded like "Thinking": its name, the skill file on expand. */
function SkillFold({ skill, fileLinks }: { skill: SkillBlock; fileLinks: FileLinks }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="my-1 text-xs text-ink-muted">
      <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)} className="flex items-center gap-1.5 rounded px-1 py-0.5 hover:text-ink focus-visible:outline focus-visible:outline-1">
        <Sparkles size={12} aria-hidden="true" />
        <ChevronRight size={12} aria-hidden="true" className={open ? "rotate-90 transition-transform" : "transition-transform"} />
        Skill · {skill.name}
      </button>
      {open && <div className="mt-1 pl-5"><LaTeXMarkdown content={skill.body} fileLinks={fileLinks} /></div>}
    </div>
  )
}

interface PiTurnProps {
  messages: AgentMessage[]
  turn: Turn
  /** The streaming reply, for the newest turn only. */
  streaming: AssistantMessage | null
  confirmed: Confirmed
  runs: Record<string, ToolRun>
  live: boolean
  expanded: boolean
  onToggle: (start: number, expanded: boolean) => void
  /** Branch keeping `entryId`; `prompt` goes into the branch's composer (Edit). */
  onBranch?: (entryId: string, prompt?: string) => void
  fileLinks: FileLinks
  onResizeStart?: ResizeStart
}

/** One turn; memoized so a streamed delta re-renders the newest turn only. */
const PiTurn = memo(function PiTurn({ messages, turn, streaming, confirmed, runs, live, expanded, onToggle, onBranch, fileLinks, onResizeStart }: PiTurnProps) {
  const prompt = messages[turn.start]
  const { text, compaction, skill } = promptOf(prompt)
  const pieces = answerPieces(messages, turn, streaming, confirmed, runs)
  const promptMarks = marksAt(confirmed, turn.start)
  const promptEntryId = turn.start < confirmed.count ? confirmed.entryIds[turn.start] : undefined
  const icon = compaction ? <FoldVertical size={14} aria-hidden="true" /> : undefined
  const answerText = pieces.flatMap((piece) => (piece.kind === "text" ? [piece.text] : [])).join("\n\n")
  const toggle = () => onToggle(turn.start, expanded)

  // Collapsed like a regular chat's pair: the header over a preview of the answer, activity folded as when expanded (omp-web's preview)
  if (!expanded) {
    return (
      <TurnCard expanded={false}>
        <TurnHeader label={compaction ? "Compaction" : clip(text) || (skill ? `/skill:${skill.name}` : "(empty)")} icon={icon} onExpand={toggle} />
        <TurnPreview tools={[]}>{pieces.length > 0 ? <Answer pieces={pieces} runs={runs} live={false} fileLinks={fileLinks} /> : null}</TurnPreview>
      </TurnCard>
    )
  }
  // Branch from the turn's last persisted entry; a turn with none yet (live, optimistic) has no branch point
  const lastConfirmed = Math.min(turn.end, confirmed.count) - 1
  const branchFrom = !live && onBranch && lastConfirmed >= turn.start ? confirmed.entryIds[lastConfirmed] : undefined
  // Edit branches from the entry before the prompt, with the prompt back in the composer
  const editFrom = !live && !compaction && onBranch && turn.start > 0 && turn.start < confirmed.count ? confirmed.entryIds[turn.start - 1] : undefined
  const questionActions = live ? [] : [
    <CopyAction key="copy" text={text} label="Copy message" />,
    ...(editFrom ? [<TurnAction key="edit" label="Edit in a new branch" icon={<CornerUpLeft size={14} aria-hidden="true" />} onClick={() => onBranch!(editFrom, text)} />] : []),
  ]
  const answerActions = live ? [] : [
    ...(answerText ? [<CopyAction key="copy" text={answerText} label="Copy message" />] : []),
    ...(answerText.trim() && typeof speechSynthesis !== "undefined" ? [<ReadAloudAction key="read" text={answerText} />] : []),
    ...(branchFrom ? [<ForkAction key="branch" label="Branch session from this point" onFork={() => onBranch!(branchFrom)} />] : []),
  ]
  return (
    <div className="relative">
    <ResizeEdges onResizeStart={onResizeStart} />
    <TurnCard expanded>
      <TurnRegion role="question" icon={icon} onCollapse={live ? undefined : toggle} actions={questionActions.length > 0 ? questionActions : undefined}>
        {promptMarks && <MarksLine marks={promptMarks} />}
        {skill && <SkillFold skill={skill} fileLinks={fileLinks} />}
        {text || !skill ? <div data-entry-id={promptEntryId}><LaTeXMarkdown content={text} fileLinks={fileLinks} /></div> : null}
      </TurnRegion>
      {pieces.length > 0 && (
        <TurnSurface>
          <TurnRegion role="answer" tail={isActivity(pieces[pieces.length - 1]) ? "activity" : "text"} actions={answerActions.length > 0 ? answerActions : undefined}>
            <Answer pieces={pieces} runs={runs} live={live} fileLinks={fileLinks} />
          </TurnRegion>
        </TurnSurface>
      )}
    </TurnCard>
    </div>
  )
})

/**
 * Every turn as the regular chat's Q&A card: the question with Edit, the answer with activity folded into "Process details" and Read aloud.
 * The newest is expanded, older ones collapse to their question over a preview of the answer until opened, in the regular chat's column
 * (its width, wide mode from the top bar's toggle). File links open through `fileLinks`.
 *
 * - Follows the end while the reader is there, `paddingBottom` keeps the last turn clear of the floating composer
 * - Mounts the newest `TURN_PAGE_SIZE` turns, earlier pages on scrolling up (or "Show earlier turns"); while the reader is away
 *   from the end the window's top stays put, so appended turns do not push the read ones out
 * - A search hit (`search`) widens the mounted window and opens its turn when needed, scrolls to the hit, centred, and holds the view
 *   there; Enter and Shift+Enter (outside text fields) step through the session's hits, Escape clears
 * - `onBranch` adds "Branch session from this point" to settled turns, with the entry id of the turn's last persisted message,
 *   and Edit, with the entry before the prompt and the prompt
 * - `onColumnChange` gets the turn column, for the top bar's model name to make way for it like a chat's
 */
export function PiTranscript({ transcript, held = [], onRetryHeld, onDeleteHeld, fileLinks, search = null, onBranch, paddingBottom, onColumnChange }: {
  transcript: Transcript
  /** Sent but unconfirmed messages, shown below the turns. */
  held?: readonly HeldMessage[]
  onRetryHeld?: (id: string) => void
  onDeleteHeld?: (id: string) => void
  fileLinks: FileLinks
  /** The session search hit to show: its turn is mounted and opened, the view scrolls to it and every hit is painted (CSS Custom Highlight API). */
  search?: PiSearchFocus | null
  onBranch?: (entryId: string, prompt?: string) => void
  paddingBottom?: number
  onColumnChange?: (column: HTMLDivElement | null) => void
}) {
  const { messages: confirmedMessages, unconfirmed, optimistic, entryIds, marks, streaming, running, runs: liveRuns } = transcript
  // Derived from the fields that change per message, not per streamed delta, so settled turns keep their props
  const messages = useMemo(() => shownMessages({ messages: confirmedMessages, unconfirmed, optimistic }), [confirmedMessages, unconfirmed, optimistic])
  const turns = useMemo(() => turnsOf(messages), [messages])
  const runs = useMemo(() => toolRunsOf({ messages: confirmedMessages, unconfirmed, optimistic, runs: liveRuns }), [confirmedMessages, unconfirmed, optimistic, liveRuns])
  const confirmed = useMemo<Confirmed>(() => ({ count: confirmedMessages.length, entryIds, marks }), [confirmedMessages.length, entryIds, marks])
  const [toggled, setToggled] = useState<Record<number, boolean>>({})
  const [pageCount, setPageCount] = useState(TURN_PAGE_SIZE)
  /** First mounted turn while the reader is away from the end, null while following the end. */
  const [anchor, setAnchor] = useState<number | null>(null)
  const [wide, setWide] = useState(false)
  const [preview, setPreview] = useState<Attachment | null>(null)
  const { width: columnWidth, startResize } = useColumnWidth()
  const scrollRef = useRef<HTMLDivElement>(null)
  const earlierRef = useRef<HTMLButtonElement>(null)
  const pinnedRef = useRef(true)
  /** Distance from the content's end to the viewport top before a page is prepended, restored after. */
  const restoreRef = useRef<number | null>(null)

  const endAnchored = Math.max(0, turns.length - pageCount)
  const start = Math.min(anchor ?? endAnchored, Math.max(0, turns.length - 1))

  useLayoutEffect(() => {
    const element = scrollRef.current
    if (!element) return
    if (restoreRef.current !== null) {
      element.scrollTop = element.scrollHeight - restoreRef.current
      restoreRef.current = null
    } else if (pinnedRef.current) {
      element.scrollTop = element.scrollHeight
    }
  })

  const showEarlier = useCallback(() => {
    const element = scrollRef.current
    if (element) restoreRef.current = element.scrollHeight - element.scrollTop
    if (anchor === null) setPageCount((count) => count + TURN_PAGE_SIZE)
    else setAnchor(Math.max(0, anchor - TURN_PAGE_SIZE))
  }, [anchor])

  useEffect(() => {
    const button = earlierRef.current
    const root = scrollRef.current
    if (!button || !root || typeof IntersectionObserver === "undefined") return
    const observer = new IntersectionObserver((entries) => {
      // Only on a real upward scroll: at scrollTop 0 on open it would race the first scroll to the end
      if (entries[0]?.isIntersecting && root.scrollTop > 0) showEarlier()
    }, { root, rootMargin: LOAD_EARLIER_ROOT_MARGIN, threshold: 0 })
    observer.observe(button)
    return () => observer.disconnect()
  }, [showEarlier, start])

  const onScroll = () => {
    const element = scrollRef.current
    if (!element) return
    const pinned = element.scrollHeight - element.scrollTop - element.clientHeight < PINNED_SLACK_PX
    pinnedRef.current = pinned
    if (pinned && anchor !== null) setAnchor(null)
    if (!pinned && anchor === null) setAnchor(start)
  }

  // The jump to a search hit: its turn is mounted and opened while rendering, then the effect scrolls to its text
  const jumped = useRef(0)
  const tabActive = useTabActive()
  useSearchHighlights(scrollRef, search)
  const hitTurn = (hit: PiSearchFocus["matches"][number]) => {
    const message = entryIds.indexOf(hit.entryId)
    return message < 0 || message >= confirmedMessages.length ? -1 : turns.findIndex((turn) => turn.start <= message && message < turn.end)
  }
  // Waits for the session's entries: nothing is loaded yet when a pick opens it
  useResetOnChange(search && confirmedMessages.length > 0 ? search.seq : null, () => {
    const turnIndex = search ? hitTurn(search.matches[search.index]) : -1
    if (turnIndex < 0) return
    const turnStart = turns[turnIndex].start
    if (turnIndex < start) setAnchor(turnIndex)
    setToggled((current) => (current[turnStart] ? current : { ...current, [turnStart]: true }))
  })
  useEffect(() => {
    const root = scrollRef.current
    const hit = search?.matches[search.index]
    if (!search || !hit || !root || jumped.current === search.seq || confirmedMessages.length === 0) return
    jumped.current = search.seq
    const range = currentRange(root, search)
    // Else the entry's text, e.g. when a hit's ordinal is past what rendered (a message too large to render shows less)
    const target = range?.startContainer.parentElement ?? root.querySelector(`${ENTRY_TEXT}[data-entry-id="${CSS.escape(hit.entryId)}"]`)
    if (!target) return
    pinnedRef.current = false
    // Instant: a smooth scroll would race the sentinel that loads the earlier page
    target.scrollIntoView({ block: "center", behavior: "instant" })
    if (range) {
      const rect = range.getBoundingClientRect()
      const view = root.getBoundingClientRect()
      root.scrollTo({ top: root.scrollTop + rect.top + rect.height / 2 - (view.top + view.height / 2), behavior: "instant" })
    }
  }, [search, confirmedMessages.length])
  // Enter and Shift+Enter step, Escape clears: from the page or the transcript, never from a text field or a button
  useEventListener("window", "keydown", (event) => {
    const target = event.target
    if (target instanceof Element && (!(target === document.body || scrollRef.current?.contains(target)) || target.closest("input, textarea, select, button, [contenteditable='true']"))) return
    if (event.key === "Escape") clearPiSearchHit()
    else if (event.key === "Enter") stepPiSearchHit(event.shiftKey ? -1 : 1)
    else return
    event.preventDefault()
  }, { active: tabActive && search !== null })

  const onToggle = useCallback((turnStart: number, expanded: boolean) => setToggled((current) => ({ ...current, [turnStart]: !expanded })), [])
  const liveTail = running || streaming !== null || optimistic !== null

  return (
    <PreviewContext.Provider value={setPreview}>
    <div ref={scrollRef} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto" style={{ paddingBottom }}>
      {search && <PiSearchBar search={search} />}
      <div ref={onColumnChange} data-chat-column className="mx-auto" style={{ maxWidth: wide ? undefined : columnWidth }}>
        {start > 0 && (
          <button ref={earlierRef} type="button" onClick={showEarlier} className="mx-auto mt-3 block rounded px-2 py-1 text-xs text-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-1">
            Show earlier turns
          </button>
        )}
        {turns.slice(start).map((turn, offset) => {
          const newest = start + offset === turns.length - 1
          const live = newest && liveTail
          return (
            <PiTurn
              key={turn.start}
              messages={messages}
              turn={turn}
              streaming={newest ? streaming : null}
              confirmed={confirmed}
              runs={runs}
              live={live}
              expanded={live || (toggled[turn.start] ?? newest)}
              onToggle={onToggle}
              onBranch={onBranch}
              fileLinks={fileLinks}
              onResizeStart={wide ? undefined : startResize}
            />
          )
        })}
        <HeldMessages held={held} onRetry={(id) => onRetryHeld?.(id)} onDelete={(id) => onDeleteHeld?.(id)} />
      </div>
      <WideModeToggle wide={wide} onToggle={() => setWide((current) => !current)} />
      <AttachmentPreview attachment={preview} onClose={() => setPreview(null)} />
    </div>
    </PreviewContext.Provider>
  )
}
