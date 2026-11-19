"use client"

import { Fragment, memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { Brain, ChevronRight, FoldVertical, TerminalSquare } from "lucide-react"
import { CopyAction, ForkAction, TurnCard, TurnHeader, TurnPreview, TurnRegion, TurnSurface, type TurnPreviewTool } from "@gigachad/chat-ui"
import "@gigachad/chat-ui/turn.css"
import { LaTeXMarkdown, type FileLinks } from "@/components/content/LaTeXMarkdown"
import { resolveLocalFileHref } from "@/lib/fileLinks"
import { renderCustomMessage, renderToolCall, toolPlacement } from "./renderers"
import { ProcessDetails } from "./tools/ProcessDetails"
import { ToolCallRow } from "./tools/ToolCallRow"
import { shownMessages, textOf, toolRunsOf, turnsOf, type MessageMarks, type Transcript, type Turn } from "./transcript"
import type { AgentMessage, AssistantMessage, BashExecutionMessage, CompactionSummaryMessage, CustomMessage, ToolCall, ToolRun } from "./types"

const MAX_LABEL_LENGTH = 80
/** Turns mounted at first and added per "Show earlier turns": older ones create no elements (smooth streaming). */
export const TURN_PAGE_SIZE = 20
/** Distance from the end (px) that still counts as reading the end, which the view then follows. */
const PINNED_SLACK_PX = 48
/** Load the earlier page while the button is still this far below the top edge, so an upward scroll continues into it. */
const LOAD_EARLIER_ROOT_MARGIN = "400px 0px 0px 0px"

/** One rendered part of an answer, in the order Pi produced it. */
type Piece =
  | { kind: "text"; key: string; text: string }
  | { kind: "thinking"; key: string; text: string; messageKey: string }
  | { kind: "tool"; key: string; call: ToolCall; messageKey: string }
  | { kind: "bash"; key: string; message: BashExecutionMessage }
  | { kind: "custom"; key: string; message: CustomMessage }
  | { kind: "mark"; key: string; marks: MessageMarks }
  | { kind: "error"; key: string; text: string }

/** Folds into "Process details" when settled; a tool with a `standalone` renderer stays between the texts. */
function isActivity(piece: Piece): boolean {
  return piece.kind === "thinking" || piece.kind === "bash" || (piece.kind === "tool" && toolPlacement(piece.call.name) === "inline")
}

function assistantPieces(message: AssistantMessage, messageKey: string): Piece[] {
  const pieces = message.content.map((block, index): Piece | null => {
    const key = `${messageKey}-${index}`
    if (block.type === "text") return block.text.trim() ? { kind: "text", key, text: block.text } : null
    if (block.type === "thinking") return block.thinking.trim() ? { kind: "thinking", key, text: block.thinking, messageKey } : null
    return { kind: "tool", key, call: block, messageKey }
  }).filter((piece): piece is Piece => piece !== null)
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

/** The answer's pieces: committed messages after the prompt (each edited one after its marks), then the streaming reply when live. */
function answerPieces(messages: AgentMessage[], turn: Turn, streaming: AssistantMessage | null, confirmed: Confirmed): Piece[] {
  const pieces: Piece[] = []
  for (let index = turn.start + 1; index < turn.end; index++) {
    const message = messages[index]
    const key = `m${index}`
    const visible = message.role === "assistant" || message.role === "bashExecution" || (message.role === "custom" && (message as CustomMessage).display)
    const marks = visible ? marksAt(confirmed, index) : undefined
    if (marks) pieces.push({ kind: "mark", key: `${key}-mark`, marks })
    if (message.role === "assistant") pieces.push(...assistantPieces(message as AssistantMessage, key))
    if (message.role === "bashExecution") pieces.push({ kind: "bash", key, message: message as BashExecutionMessage })
    if (message.role === "custom" && (message as CustomMessage).display) pieces.push({ kind: "custom", key, message: message as CustomMessage })
  }
  if (streaming) pieces.push(...assistantPieces(streaming, "streaming"))
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

function BashBlock({ message }: { message: BashExecutionMessage }) {
  return (
    <div className="my-1 font-mono text-xs">
      <div className="flex items-center gap-1.5 text-ink-muted"><TerminalSquare size={12} aria-hidden="true" />$ {message.command}</div>
      {message.output && <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap text-ink-subtle">{message.output}</pre>}
    </div>
  )
}

/** Quiet line naming a message's label and `context_edit`. */
function MarksLine({ marks }: { marks: MessageMarks }) {
  const parts = [marks.label, marks.context === "omitted" ? "Omitted from context" : marks.context === "replaced" ? "Replaced in context" : undefined].filter(Boolean)
  return <p className="my-1 text-xs text-ink-faint">{parts.join(" · ")}</p>
}

function renderPiece(piece: Piece, runs: Record<string, ToolRun>, live: boolean, fileLinks: FileLinks): ReactNode {
  switch (piece.kind) {
    case "text": return <LaTeXMarkdown key={piece.key} content={piece.text} streaming={live && piece.key.startsWith("streaming")} fileLinks={fileLinks} />
    case "thinking": return <ThinkingBlock key={piece.key} text={piece.text} />
    case "tool": {
      const run = runs[piece.call.id]
      const row = () => <ToolCallRow call={piece.call} run={run} streaming={live} defaultCollapsed onOpenFile={(path) => fileLinks.open(resolveLocalFileHref(path, fileLinks.base) ?? path)} />
      return <Fragment key={piece.key}>{renderToolCall(piece.call, run, row)}</Fragment>
    }
    case "bash": return <BashBlock key={piece.key} message={piece.message} />
    case "custom": {
      const text = () => <LaTeXMarkdown content={textOf(piece.message.content)} fileLinks={fileLinks} />
      return <Fragment key={piece.key}>{renderCustomMessage(piece.message, text)}</Fragment>
    }
    case "mark": return <MarksLine key={piece.key} marks={piece.marks} />
    case "error": return <p key={piece.key} className="my-1 text-xs text-danger">{piece.text}</p>
  }
}

/**
 * The answer like omp-web shows it: text stays visible, every run of activity (thinking, tool calls, shell commands) between
 * texts folds into "Process details". A live turn stays flat so tool progress is in view, it folds once settled.
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

function clip(text: string): string {
  const firstLine = text.trim().split("\n")[0] ?? ""
  return firstLine.length <= MAX_LABEL_LENGTH ? firstLine : `${firstLine.slice(0, MAX_LABEL_LENGTH)}\u2026`
}

function promptOf(message: AgentMessage): { text: string; compaction: boolean } {
  if (message.role === "compactionSummary") return { text: (message as CompactionSummaryMessage).summary, compaction: true }
  return { text: textOf((message as { content: string | { type: string; text?: string }[] }).content), compaction: false }
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
  onBranch?: (entryId: string) => void
  fileLinks: FileLinks
}

/** One turn; memoized so a streamed delta re-renders the newest turn only. */
const PiTurn = memo(function PiTurn({ messages, turn, streaming, confirmed, runs, live, expanded, onToggle, onBranch, fileLinks }: PiTurnProps) {
  const prompt = messages[turn.start]
  const { text, compaction } = promptOf(prompt)
  const pieces = answerPieces(messages, turn, streaming, confirmed)
  const promptMarks = marksAt(confirmed, turn.start)
  const icon = compaction ? <FoldVertical size={14} aria-hidden="true" /> : undefined
  const answerText = pieces.flatMap((piece) => (piece.kind === "text" ? [piece.text] : [])).join("\n\n")
  const toggle = () => onToggle(turn.start, expanded)

  if (!expanded) {
    const tools: TurnPreviewTool[] = pieces.flatMap((piece) => (piece.kind === "tool" ? [{ key: piece.key, label: piece.call.name }] : []))
    const lastText = pieces.findLast((piece) => piece.kind === "text")
    return (
      <TurnCard expanded={false}>
        <TurnHeader label={compaction ? "Compaction" : clip(text) || "(empty)"} icon={icon} onExpand={toggle} />
        <TurnPreview tools={tools}>{lastText?.kind === "text" ? <LaTeXMarkdown content={lastText.text} compact /> : null}</TurnPreview>
      </TurnCard>
    )
  }
  // Branch from the turn's last persisted entry; a turn with none yet (live, optimistic) has no branch point
  const lastConfirmed = Math.min(turn.end, confirmed.count) - 1
  const branchFrom = !live && onBranch && lastConfirmed >= turn.start ? confirmed.entryIds[lastConfirmed] : undefined
  const answerActions = live ? [] : [
    ...(answerText ? [<CopyAction key="copy" text={answerText} label="Copy answer as markdown" />] : []),
    ...(branchFrom ? [<ForkAction key="branch" label="Branch session from this point" onFork={() => onBranch!(branchFrom)} />] : []),
  ]
  return (
    <TurnCard expanded>
      <TurnRegion role="question" icon={icon} onCollapse={live ? undefined : toggle} actions={live ? undefined : <CopyAction text={text} label="Copy question as markdown" />}>
        {promptMarks && <MarksLine marks={promptMarks} />}
        {compaction ? <p className="text-xs text-ink-muted">Compacted: {clip(text)}</p> : <p className="whitespace-pre-wrap text-sm leading-6">{text}</p>}
      </TurnRegion>
      {pieces.length > 0 && (
        <TurnSurface>
          <TurnRegion role="answer" tail={isActivity(pieces[pieces.length - 1]) ? "activity" : "text"} actions={answerActions.length > 0 ? answerActions : undefined}>
            <Answer pieces={pieces} runs={runs} live={live} fileLinks={fileLinks} />
          </TurnRegion>
        </TurnSurface>
      )}
    </TurnCard>
  )
})

/**
 * Every turn as a gigachad Q&A card, the newest expanded, older ones collapsed until opened. File links open through `fileLinks`.
 *
 * - Follows the end while the reader is there
 * - Mounts the newest `TURN_PAGE_SIZE` turns, earlier pages on scrolling up (or "Show earlier turns"); while the reader is away
 *   from the end the window's top stays put, so appended turns do not push the read ones out
 * - `onBranch` adds "Branch session from this point" to settled turns, with the entry id of the turn's last persisted message
 */
export function PiTranscript({ transcript, fileLinks, onBranch }: { transcript: Transcript; fileLinks: FileLinks; onBranch?: (entryId: string) => void }) {
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

  const onToggle = useCallback((turnStart: number, expanded: boolean) => setToggled((current) => ({ ...current, [turnStart]: !expanded })), [])
  const liveTail = running || streaming !== null || optimistic !== null

  return (
    <div ref={scrollRef} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-3 px-4 py-6">
        {turns.length === 0 && <p className="py-16 text-center text-sm text-ink-faint">New session</p>}
        {start > 0 && (
          <button ref={earlierRef} type="button" onClick={showEarlier} className="self-center rounded px-2 py-1 text-xs text-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-1">
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
            />
          )
        })}
      </div>
    </div>
  )
}
