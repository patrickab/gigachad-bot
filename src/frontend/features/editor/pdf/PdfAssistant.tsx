"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { SendStopButton, useAssistantChat, useOneShotStream } from "@/components/content/AssistantChat"
import { ChatSidebar } from "@/components/ui/ChatSidebar"
import { SelectionBubble } from "@/components/ui/SelectionBubble"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useEventListener } from "@/hooks/useEventListener"
import { attachReference } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"

/** Poll interval while MinerU extracts the PDF. */
const EXTRACTION_POLL_MS = 5000
const CARD_WIDTH_PX = 380
const CARD_GAP_PX = 8
/** Room a card needs below the selection before it flips above it. */
const CARD_MIN_ROOM_PX = 260

type PdfMarkdown =
  | { status: "loading" | "extracting" }
  | { status: "ready"; markdown: string }
  | { status: "error"; message: string }

const STATUS_MESSAGES: Record<"loading" | "extracting", string> = {
  loading: "Loading document text…",
  extracting: "Extracting document text…",
}

const SYS_DOCUMENT = (markdown: string) =>
  `You answer questions about the PDF document below, extracted to Markdown. Base answers on the document and say when it does not cover something. Use Markdown, and LaTeX for math.\n\n<document>\n${markdown}\n</document>`

const SYS_SELECTION = (markdown: string, selection: string, page: number) =>
  `${SYS_DOCUMENT(markdown)}\n\nThe reader selected this passage on page ${page}:\n\n<selection>\n${selection}\n</selection>\n\nAnswer the reader's question about the selection briefly.`

/** Only mounted files and library PDFs carry a MinerU extraction. */
function hasExtraction(path: string): boolean {
  return path.startsWith("/") || path.startsWith("PDFs/")
}

/** MinerU Markdown of the PDF at `path`, polled while extraction runs. Owners key by `path`, so it never changes. */
function usePdfMarkdown(path: string): PdfMarkdown {
  const [state, setState] = useState<PdfMarkdown>(() =>
    hasExtraction(path) ? { status: "loading" } : { status: "error", message: "This PDF has no text extraction." })

  useEffect(() => {
    if (!hasExtraction(path)) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      try {
        const attachment = await attachReference(path)
        if (cancelled) return
        if (attachment.parsedMd) {
          setState({ status: "ready", markdown: attachment.parsedMd })
          return
        }
        setState({ status: "extracting" })
        timer = setTimeout(load, EXTRACTION_POLL_MS)
      } catch (error) {
        if (!cancelled) setState({ status: "error", message: errorMessage(error, "Could not load the document text.") })
      }
    }
    void load()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [path])

  return state
}

function DocumentStatus({ pdfText }: { pdfText: PdfMarkdown }) {
  if (pdfText.status === "ready") return null
  const message = pdfText.status === "error" ? pdfText.message : STATUS_MESSAGES[pdfText.status]
  return <div role="status" className="px-4 py-2 text-[11px] text-ink-subtle">{message}</div>
}

/**
 * Wrap a PDF viewer with an LLM assistant grounded in the PDF's MinerU Markdown.
 *
 * - `chatOpen` shows a `ChatSidebar` right of the viewer, under a top bar `topInset` high. Its history lives as long as this component
 * - Selecting text opens a one-shot question card, closed by Escape or any click outside it
 */
export function PdfAssistant({ model, path, chatOpen, onChatOpenChange, topInset, children }: {
  path: string
  model: string
  chatOpen: boolean
  onChatOpenChange: (open: boolean) => void
  topInset: number
  children: ReactNode
}) {
  const pdfText = usePdfMarkdown(path)
  const markdown = pdfText.status === "ready" ? pdfText.markdown : ""
  const chat = useAssistantChat(model, SYS_DOCUMENT(markdown))
  const { containerRef, selected, clear: clearSelection } = usePdfSelection()
  const [asking, setAsking] = useState<PdfSelection | null>(null)

  return (
    <ChatSidebar
      open={chatOpen}
      onOpenChange={onChatOpenChange}
      chat={chat}
      inputLabel="Ask about this PDF"
      disabled={pdfText.status !== "ready"}
      status={<DocumentStatus pdfText={pdfText} />}
      topInset={topInset}
    >
      {/* No scrollbar between the pages and the open chat, wheel and keys still scroll. Important: the global `* { scrollbar-width }` is unlayered */}
      <div ref={containerRef} className={chatOpen ? "h-full [&_.pdf-scroll]:[scrollbar-width:none]!" : "h-full"}>
        {children}
      </div>
      {selected && (
        <SelectionBubble
          bounds={selected.anchor}
          label="Ask about selection"
          onActivate={() => { setAsking(selected); clearSelection() }}
        />
      )}
      {/* Portaled: native key and pointer events inside the card must not reach the selection listeners */}
      {asking && createPortal(
        <SelectionQuestionCard
          key={asking.id}
          selection={asking}
          pdfText={pdfText}
          model={model}
          onClose={() => setAsking(null)}
        />,
        document.body,
      )}
    </ChatSidebar>
  )
}

interface PdfSelection {
  id: number
  text: string
  page: number
  /** Bounds of the selection's last line, the bubble's tip touches its lower-left corner */
  anchor: DOMRect
}

/**
 * The page text selection made inside `containerRef`, set on pointer or key release.
 *
 * - Only a selection changed since the last capture counts, a click that keeps it (scrollbar) does not
 * - Cleared when the selection collapses, and on scroll or resize, which would strand the bubble
 */
function usePdfSelection() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [selected, setSelected] = useState<PdfSelection | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let changed = false
    let nextId = 0
    const onSelectionChange = () => {
      changed = true
      if (window.getSelection()?.isCollapsed) setSelected(null)
    }
    const capture = () => {
      if (!changed) return
      changed = false
      const found = readPageSelection(container)
      if (found) setSelected({ ...found, id: nextId++ })
    }
    const dismiss = () => setSelected(null)
    document.addEventListener("selectionchange", onSelectionChange)
    container.addEventListener("pointerup", capture)
    container.addEventListener("keyup", capture)
    window.addEventListener("scroll", dismiss, true)
    window.addEventListener("resize", dismiss)
    return () => {
      document.removeEventListener("selectionchange", onSelectionChange)
      container.removeEventListener("pointerup", capture)
      container.removeEventListener("keyup", capture)
      window.removeEventListener("scroll", dismiss, true)
      window.removeEventListener("resize", dismiss)
    }
  }, [])

  const clear = useCallback(() => setSelected(null), [])
  return { containerRef, selected, clear }
}

/** Non-empty selection starting on a PDF page inside `container`, with its page number and last line's bounds. */
function readPageSelection(container: HTMLElement): Omit<PdfSelection, "id"> | null {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null
  const text = selection.toString().trim()
  const anchor = selection.anchorNode
  const anchorElement = anchor instanceof Element ? anchor : anchor?.parentElement
  const page = anchorElement?.closest<HTMLElement>(".pdf-page[data-page-number]")
  if (!text || !page || !container.contains(page)) return null
  const range = selection.getRangeAt(0)
  const lineRects = [...range.getClientRects()].filter((rect) => rect.width > 0 && rect.height > 0)
  return { text, page: Number(page.dataset.pageNumber), anchor: lineRects.at(-1) ?? range.getBoundingClientRect() }
}

/** Below the selection, or above it when the viewport has too little room below. */
function cardPosition(anchor: DOMRect): CSSProperties {
  const left = Math.min(Math.max(anchor.left, CARD_GAP_PX), window.innerWidth - CARD_WIDTH_PX - CARD_GAP_PX)
  const roomBelow = window.innerHeight - anchor.bottom
  return roomBelow >= CARD_MIN_ROOM_PX || roomBelow >= anchor.top
    ? { left, top: anchor.bottom + CARD_GAP_PX, maxHeight: roomBelow - 2 * CARD_GAP_PX }
    : { left, bottom: window.innerHeight - anchor.top + CARD_GAP_PX, maxHeight: anchor.top - 2 * CARD_GAP_PX }
}

function isCopyShortcut(event: KeyboardEvent<HTMLInputElement>): boolean {
  return (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "c"
}

/** Stateless question about one selection: every question is a fresh call with the document and the selection. */
function SelectionQuestionCard({ selection, pdfText, model, onClose }: {
  selection: PdfSelection
  pdfText: PdfMarkdown
  model: string
  onClose: () => void
}) {
  const cardRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const [question, setQuestion] = useState("")
  const { text: answer, isStreaming, ask: askModel, abort, stop } = useOneShotStream(model, { appendErrors: true })
  const ready = pdfText.status === "ready"

  const close = useCallback(() => { abort(); onClose() }, [abort, onClose])
  useEscapeLayer(close)
  useLayoutEffect(() => { inputRef.current?.focus() }, [])

  // Any press outside the card closes it, including the press that starts the next selection
  useEventListener("window", "pointerdown", (event) => {
    if (!cardRef.current?.contains(event.target as Node)) close()
  }, { capture: true })

  const ask = async () => {
    if (pdfText.status !== "ready") return
    await askModel(question, SYS_SELECTION(pdfText.markdown, selection.text, selection.page))
  }

  // Focus moved into the card, so copy still copies the page selection while the question is empty
  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") { event.preventDefault(); void ask() }
    else if (isCopyShortcut(event) && !question) { event.preventDefault(); void navigator.clipboard.writeText(selection.text) }
  }

  return (
    <div
      ref={cardRef}
      role="dialog"
      aria-label={`Ask about selection on page ${selection.page}`}
      style={{ ...cardPosition(selection.anchor), width: CARD_WIDTH_PX }}
      className="fixed z-50 flex flex-col overflow-hidden rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]"
    >
      <div className="shrink-0 border-b border-divider/50 px-3 py-2 text-[11px] text-ink-subtle">
        <span className="line-clamp-2">p. {selection.page} · {selection.text}</span>
      </div>
      {answer && (
        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2 text-[11px] leading-relaxed">
          <LaTeXMarkdown content={answer} />
        </div>
      )}
      {!ready && <DocumentStatus pdfText={pdfText} />}
      <div className="flex shrink-0 items-center gap-2 border-t border-divider/50 px-3 py-1.5">
        <input
          ref={inputRef}
          aria-label="Question about the selection"
          value={question}
          disabled={!ready}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={onKeyDown}
          className="flex-1 bg-transparent text-xs text-ink outline-none disabled:opacity-50"
        />
        <SendStopButton
          isStreaming={isStreaming}
          onStop={stop}
          onSend={() => void ask()}
          disabled={!question.trim() || !ready}
          sendLabel="Ask"
          className="shrink-0 p-1 text-ink-subtle hover:text-ink"
          iconClassName="h-3 w-3"
        />
      </div>
    </div>
  )
}
