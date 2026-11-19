"use client"

import { useEffect, useRef, useState, type ReactNode } from "react"
import { Brain, ChevronRight, FoldVertical, TerminalSquare } from "lucide-react"
import { CopyAction, TurnCard, TurnHeader, TurnPreview, TurnRegion, TurnSurface, type TurnPreviewTool } from "@gigachad/chat-ui"
import "@gigachad/chat-ui/turn.css"
import { LaTeXMarkdown, type FileLinks } from "@/components/content/LaTeXMarkdown"
import { resolveLocalFileHref } from "@/lib/fileLinks"
import { ProcessDetails } from "./tools/ProcessDetails"
import { ToolCallRow } from "./tools/ToolCallRow"
import { textOf, toolRunsOf, turnsOf, type Transcript, type Turn } from "./transcript"
import type { AgentMessage, AssistantMessage, BashExecutionMessage, CompactionSummaryMessage, CustomMessage, ToolCall, ToolRun } from "./types"

const MAX_LABEL_LENGTH = 80

/** One rendered part of an answer, in the order Pi produced it. */
type Piece =
  | { kind: "text"; key: string; text: string }
  | { kind: "thinking"; key: string; text: string; messageKey: string }
  | { kind: "tool"; key: string; call: ToolCall; messageKey: string }
  | { kind: "bash"; key: string; message: BashExecutionMessage }
  | { kind: "error"; key: string; text: string }

function isActivity(piece: Piece): boolean {
  return piece.kind === "thinking" || piece.kind === "tool" || piece.kind === "bash"
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

/** The answer's pieces: committed messages after the prompt, then the streaming reply when live. */
function answerPieces(messages: AgentMessage[], turn: Turn, streaming: AssistantMessage | null): Piece[] {
  const pieces: Piece[] = []
  for (let index = turn.start + 1; index < turn.end; index++) {
    const message = messages[index]
    if (message.role === "assistant") pieces.push(...assistantPieces(message as AssistantMessage, `m${index}`))
    if (message.role === "bashExecution") pieces.push({ kind: "bash", key: `m${index}`, message: message as BashExecutionMessage })
    if (message.role === "custom" && (message as CustomMessage).display) pieces.push({ kind: "text", key: `m${index}`, text: textOf((message as CustomMessage).content) })
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

function renderPiece(piece: Piece, runs: Record<string, ToolRun>, live: boolean, fileLinks: FileLinks): ReactNode {
  switch (piece.kind) {
    case "text": return <LaTeXMarkdown key={piece.key} content={piece.text} streaming={live && piece.key.startsWith("streaming")} fileLinks={fileLinks} />
    case "thinking": return <ThinkingBlock key={piece.key} text={piece.text} />
    case "tool": return <ToolCallRow key={piece.key} call={piece.call} run={runs[piece.call.id]} streaming={live} defaultCollapsed onOpenFile={(path) => fileLinks.open(resolveLocalFileHref(path, fileLinks.base) ?? path)} />
    case "bash": return <BashBlock key={piece.key} message={piece.message} />
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

function PiTurn({ prompt, pieces, runs, live, expanded, onToggle, fileLinks }: {
  prompt: AgentMessage
  pieces: Piece[]
  runs: Record<string, ToolRun>
  live: boolean
  expanded: boolean
  onToggle: () => void
  fileLinks: FileLinks
}) {
  const { text, compaction } = promptOf(prompt)
  const icon = compaction ? <FoldVertical size={14} aria-hidden="true" /> : undefined
  const answerText = pieces.flatMap((piece) => (piece.kind === "text" ? [piece.text] : [])).join("\n\n")

  if (!expanded) {
    const tools: TurnPreviewTool[] = pieces.flatMap((piece) => (piece.kind === "tool" ? [{ key: piece.key, label: piece.call.name }] : []))
    const lastText = pieces.findLast((piece) => piece.kind === "text")
    return (
      <TurnCard expanded={false}>
        <TurnHeader label={compaction ? "Compaction" : clip(text) || "(empty)"} icon={icon} onExpand={onToggle} />
        <TurnPreview tools={tools}>{lastText?.kind === "text" ? <LaTeXMarkdown content={lastText.text} compact /> : null}</TurnPreview>
      </TurnCard>
    )
  }
  return (
    <TurnCard expanded>
      <TurnRegion role="question" icon={icon} onCollapse={live ? undefined : onToggle} actions={live ? undefined : <CopyAction text={text} label="Copy question as markdown" />}>
        {compaction ? <p className="text-xs text-ink-muted">Compacted: {clip(text)}</p> : <p className="whitespace-pre-wrap text-sm leading-6">{text}</p>}
      </TurnRegion>
      {pieces.length > 0 && (
        <TurnSurface>
          <TurnRegion role="answer" tail={isActivity(pieces[pieces.length - 1]) ? "activity" : "text"} actions={live || !answerText ? undefined : <CopyAction text={answerText} label="Copy answer as markdown" />}>
            <Answer pieces={pieces} runs={runs} live={live} fileLinks={fileLinks} />
          </TurnRegion>
        </TurnSurface>
      )}
    </TurnCard>
  )
}

/** Every turn as a gigachad Q&A card, the newest expanded, older ones collapsed until opened. Follows the end while the reader is there. File links open through `fileLinks`. */
export function PiTranscript({ transcript, fileLinks }: { transcript: Transcript; fileLinks: FileLinks }) {
  const messages = shownMessages(transcript)
  const turns = turnsOf(messages)
  const runs = toolRunsOf(transcript)
  const [toggled, setToggled] = useState<Record<number, boolean>>({})
  const scrollRef = useRef<HTMLDivElement>(null)
  const pinnedRef = useRef(true)

  useEffect(() => {
    const element = scrollRef.current
    if (element && pinnedRef.current) element.scrollTop = element.scrollHeight
  })

  const onScroll = () => {
    const element = scrollRef.current
    if (element) pinnedRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 48
  }

  return (
    <div ref={scrollRef} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-3 px-4 py-6">
        {turns.length === 0 && <p className="py-16 text-center text-sm text-ink-faint">New session</p>}
        {turns.map((turn, index) => {
          const newest = index === turns.length - 1
          const live = newest && (transcript.running || transcript.streaming !== null || transcript.optimistic !== null)
          const expanded = live || (toggled[turn.start] ?? newest)
          return (
            <PiTurn
              key={turn.start}
              prompt={messages[turn.start]}
              pieces={answerPieces(messages, turn, newest ? transcript.streaming : null)}
              runs={runs}
              live={live}
              expanded={expanded}
              onToggle={() => setToggled((current) => ({ ...current, [turn.start]: !expanded }))}
              fileLinks={fileLinks}
            />
          )
        })}
      </div>
    </div>
  )
}
