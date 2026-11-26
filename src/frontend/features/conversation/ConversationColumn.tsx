"use client"

import { Fragment, isValidElement, memo, useCallback, useEffect, useEffectEvent, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from "react"
import { TurnCard, TurnHeader } from "@gigachad/chat-ui"
import "@gigachad/chat-ui/turn.css"
import { ResizeEdges, useColumnWidth, type ResizeStart } from "@/components/ui/columnResize"
import { useTabActive } from "@/contexts/TabContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useEventListener } from "@/hooks/useEventListener"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { isModKey, isTerminalKeyEvent } from "@/lib/utils"
import { WideModeToggle } from "@/features/chat/WideModeToggle"

/** Turns mounted at first and added per "Show earlier turns": older ones create no elements (smooth streaming). */
export const TURN_PAGE_SIZE = 20
/** Distance from the end (px) that still counts as reading the end, which the view then follows. */
const PINNED_SLACK_PX = 48
/** Load the earlier page while the button is still this far below the top edge, so an upward scroll continues into it. */
const LOAD_EARLIER_ROOT_MARGIN = "400px 0px 0px 0px"
/** A collapsed turn renders its preview once it comes this close (px) to the viewport. */
const PREVIEW_MARGIN_PX = 400
/** Turns kept above the keyboard-focused one, so it sits a few rows below the top. */
const FOCUS_ROWS_ABOVE = 3

/** One question and its answer as the column sees it. */
export interface ConversationTurn {
  /** Stable for the life of the turn: it does not change when an earlier turn goes. Open state is kept by it. */
  key: string
  /** The collapsed header's one line. */
  label: string
  /** Replaces the question's avatar in the collapsed header. */
  icon?: ReactNode
  /** Streaming or about to: always open, never collapsible. */
  live: boolean
}

/** A turn to mount, open and scroll to, once per `seq`. */
export interface ConversationReveal {
  key: string
  seq: number
  /** Scrolls to a place inside the opened turn, true when it found it; default: the turn's top. */
  scroll?: (root: HTMLElement) => boolean
}

export interface ConversationColumnProps<T extends ConversationTurn> {
  turns: readonly T[]
  /** A collapsed turn's preview under its header (e.g. a `TurnPreview`), null for none. */
  renderPreview: (turn: T, index: number) => ReactNode
  /** An open turn's card content (its regions); `collapse` is unset for a live turn. */
  renderBody: (turn: T, index: number, collapse: (() => void) | undefined) => ReactNode
  /** A row after a turn, e.g. a separator. */
  renderAfter?: (turn: T, index: number) => ReactNode
  /** Above the column, inside the scroller (e.g. a search bar). */
  header?: ReactNode
  /** Below the last turn, inside the column (e.g. held messages). */
  footer?: ReactNode
  reveal?: ConversationReveal | null
  /** Each new value scrolls to the end and follows it again, e.g. on send. */
  followSeq?: number
  /** Keeps the last turn clear of a floating composer. */
  paddingBottom?: number
  /** The turn column (`[data-chat-column]`), e.g. for the top bar's label to make way for it. */
  onColumnChange?: (column: HTMLDivElement | null) => void
  /** The scroll container, for a caller that paints or listens in it. */
  scrollRef?: RefObject<HTMLDivElement | null>
  /** Wide mode held by the caller, e.g. across chats that remount the column; unset: the column holds its own. */
  wide?: boolean
  onWideChange?: (wide: boolean) => void
}

/** Same element type, key and shallowly equal props: a re-created element of a memoized turn changes nothing. */
function sameNode(a: ReactNode, b: ReactNode): boolean {
  if (Object.is(a, b)) return true
  if (!isValidElement(a) || !isValidElement(b) || a.type !== b.type || a.key !== b.key) return false
  const left = a.props as Record<string, unknown>
  const right = b.props as Record<string, unknown>
  const keys = Object.keys(left)
  return keys.length === Object.keys(right).length && keys.every((key) => Object.is(left[key], right[key]))
}

/**
 * A collapsed turn's preview, rendered once it nears the viewport (or `wake` is set): a long history then parses only what is read.
 * Until then it reserves the preview's height cap. A card in a hidden tab (no size) waits for its intersection.
 */
function LazyPreview({ wake, children }: { wake: boolean; children: ReactNode }) {
  const [near, setNear] = useState(false)
  const attach = useCallback((host: HTMLDivElement | null) => {
    if (!host) return
    // Already visible: fill before paint rather than flash the placeholder
    const rect = host.getBoundingClientRect()
    if (rect.width > 0 && rect.height > 0 && rect.bottom >= -PREVIEW_MARGIN_PX && rect.top <= window.innerHeight + PREVIEW_MARGIN_PX) {
      setNear(true)
      return
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) setNear(true)
    }, { rootMargin: `${PREVIEW_MARGIN_PX}px` })
    observer.observe(host)
    return () => observer.disconnect()
  }, [])
  if (children == null) return null
  // Without an observer nothing would wake it
  return near || wake || typeof IntersectionObserver === "undefined" ? children : <div ref={attach} data-preview-pending style={{ height: "15vh" }} />
}

interface ColumnTurnProps {
  turnKey: string
  label: string
  icon: ReactNode
  open: boolean
  /** Keyboard focus is on this turn. */
  focused: boolean
  /** The preview when collapsed, the card content when open. */
  content: ReactNode
  onExpand: () => void
  onResizeStart?: ResizeStart
}

/** One turn's card; skipped while its props hold, so a streamed frame re-renders the live turn only. */
const ColumnTurn = memo(function ColumnTurn({ turnKey, label, icon, open, focused, content, onExpand, onResizeStart }: ColumnTurnProps) {
  // Focus inside the card (its header by Tab) loads the preview, so the next Tab reaches it
  const [woken, setWoken] = useState(false)
  if (!open) {
    return (
      <div data-turn-key={turnKey} onFocus={() => setWoken(true)}>
        <TurnCard expanded={false} focused={focused}>
          <TurnHeader label={label} icon={icon} onExpand={onExpand} />
          <LazyPreview wake={woken || focused}>{content}</LazyPreview>
        </TurnCard>
      </div>
    )
  }
  return (
    <div data-turn-key={turnKey} className="relative">
      <ResizeEdges onResizeStart={onResizeStart} />
      <TurnCard expanded focused={focused}>{content}</TurnCard>
    </div>
  )
}, (prev, next) => prev.turnKey === next.turnKey && prev.label === next.label && prev.open === next.open && prev.focused === next.focused
  && prev.onExpand === next.onExpand && prev.onResizeStart === next.onResizeStart && sameNode(prev.icon, next.icon) && sameNode(prev.content, next.content))

/** The mounted card of the turn `key`. */
function turnElement(root: HTMLElement, key: string): Element | undefined {
  return [...root.querySelectorAll("[data-turn-key]")].find((candidate) => candidate.getAttribute("data-turn-key") === key)
}

/** Scrolls the turn `key` to the top of the view, true when it is mounted. */
function scrollToTurn(root: HTMLElement, key: string): boolean {
  const element = turnElement(root, key)
  element?.scrollIntoView({ block: "start", behavior: "instant" })
  return element !== undefined
}

/** Scrolls to `top`, smoothly where the browser can. */
function glideTo(element: HTMLElement, top: number) {
  if (typeof element.scrollTo === "function") element.scrollTo({ top, behavior: "smooth" })
  else element.scrollTop = top
}

/**
 * The column of a conversation's turns, shared by the chat and Pi: every turn a card, open or collapsed to its header over a preview.
 *
 * - The newest turn is open, a live one always; others open and collapse on click, kept by `turn.key`
 * - Opens at the end and follows it while the reader is there, `paddingBottom` keeps the last turn clear of the floating composer.
 *   Until the reader first scrolls, clicks or types in it, late layout (diagrams, plots, the composer) keeps it at the end.
 *   Only a new turn glides into view, a streamed frame or an opened history jumps
 * - Mounts the newest `TURN_PAGE_SIZE` turns, earlier pages on scrolling up (or "Show earlier turns"); while the reader is away
 *   from the end the window's top stays put, so appended turns do not push the read ones out
 * - A collapsed turn renders its preview near the viewport only
 * - Ctrl/Cmd+Up/Down from the column or its surface (e.g. the composer) moves a keyboard focus over the turns, Enter opens or
 *   collapses the focused one, Escape drops the focus
 * - `reveal` mounts and opens its turn, then scrolls to it (or where its `scroll` says) and holds the view there
 * - Column width (drag a card edge, shared app-wide) and wide mode (the top bar's toggle)
 */
export function ConversationColumn<T extends ConversationTurn>({ turns, renderPreview, renderBody, renderAfter, header, footer, reveal = null, followSeq = 0, paddingBottom, onColumnChange, scrollRef: outerScrollRef, wide: heldWide, onWideChange }: ConversationColumnProps<T>) {
  const [toggled, setToggled] = useState<Record<string, boolean>>({})
  const [pageCount, setPageCount] = useState(TURN_PAGE_SIZE)
  /** First mounted turn while the reader is away from the end, null while following the end. */
  const [anchor, setAnchor] = useState<number | null>(null)
  const [focusedKey, setFocusedKey] = useState<string | null>(null)
  const [ownWide, setOwnWide] = useState(false)
  const wide = heldWide ?? ownWide
  const toggleWide = () => {
    if (heldWide === undefined) setOwnWide(!ownWide)
    onWideChange?.(!wide)
  }
  const { width: columnWidth, startResize } = useColumnWidth()
  const scrollRef = useRef<HTMLDivElement>(null)
  /** The scroller into `scrollRef` and the caller's. */
  const attachScroller = useCallback((element: HTMLDivElement | null) => {
    scrollRef.current = element
    if (outerScrollRef) outerScrollRef.current = element
  }, [outerScrollRef])
  const columnRef = useRef<HTMLDivElement>(null)
  const attachColumn = useCallback((element: HTMLDivElement | null) => {
    columnRef.current = element
    onColumnChange?.(element)
  }, [onColumnChange])
  const earlierRef = useRef<HTMLButtonElement>(null)
  const pinnedRef = useRef(true)
  /** Late layout keeps the view at the end until the reader first interacts or a jump moves it. */
  const settlingRef = useRef(true)
  /** A new turn is gliding into view: the scroll events on its way do not count as the reader leaving the end. */
  const glidingRef = useRef(false)
  const turnCountRef = useRef(turns.length)
  /** Distance from the content's end to the viewport top before a page is prepended, restored after. */
  const restoreRef = useRef<number | null>(null)

  const endAnchored = Math.max(0, turns.length - pageCount)
  const start = Math.min(anchor ?? endAnchored, Math.max(0, turns.length - 1))
  const isOpen = (turn: T, index: number) => turn.live || (toggled[turn.key] ?? index === turns.length - 1)

  // Following again resets the window to the end; the ref is set before the scroll below reads it
  useResetOnChange(followSeq, () => setAnchor(null))
  useLayoutEffect(() => {
    if (followSeq) pinnedRef.current = true
  }, [followSeq])

  useLayoutEffect(() => {
    const element = scrollRef.current
    const previous = turnCountRef.current
    turnCountRef.current = turns.length
    if (!element) return
    if (restoreRef.current !== null) {
      element.scrollTop = element.scrollHeight - restoreRef.current
      restoreRef.current = null
    } else if (pinnedRef.current) {
      if (previous > 0 && turns.length > previous) {
        glidingRef.current = true
        glideTo(element, element.scrollHeight)
      } else {
        element.scrollTop = element.scrollHeight
      }
    }
  })

  // The reader's first touch ends settling and any glide
  const interact = () => {
    settlingRef.current = false
    glidingRef.current = false
  }
  useEventListener(scrollRef, "wheel", interact, { passive: true })
  useEventListener(scrollRef, "touchstart", interact, { passive: true })
  useEventListener(scrollRef, "pointerdown", interact)
  useEventListener(scrollRef, "keydown", interact)

  useEffect(() => {
    const element = scrollRef.current
    const column = columnRef.current
    if (!element || !column || typeof ResizeObserver === "undefined") return
    // Not gated on `pinnedRef`: scroll anchoring shifts the view when content above grows, which is no reader scroll
    const keepAtEnd = new ResizeObserver(() => {
      if (!settlingRef.current) return
      element.scrollTop = element.scrollHeight
      pinnedRef.current = true
    })
    keepAtEnd.observe(element)
    keepAtEnd.observe(column)
    return () => keepAtEnd.disconnect()
  }, [])

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
    if (glidingRef.current) {
      if (!pinned) return
      glidingRef.current = false
    }
    pinnedRef.current = pinned
    if (pinned && anchor !== null) setAnchor(null)
    if (!pinned && anchor === null) setAnchor(start)
  }

  // A reveal mounts and opens its turn while rendering, once its turn is there, then the effect scrolls to it, once per `seq`
  const [revealedSeq, setRevealedSeq] = useState<number | null>(null)
  if (reveal && reveal.seq !== revealedSeq) {
    const index = turns.findIndex((turn) => turn.key === reveal.key)
    if (index >= 0) {
      setRevealedSeq(reveal.seq)
      if (index < start) setAnchor(index)
      setToggled((current) => (current[reveal.key] ? current : { ...current, [reveal.key]: true }))
    }
  }
  const scrolledSeq = useRef<number | null>(null)
  useEffect(() => {
    const root = scrollRef.current
    if (!reveal || !root || revealedSeq !== reveal.seq || scrolledSeq.current === reveal.seq) return
    scrolledSeq.current = reveal.seq
    interact()
    // Held where it lands: a scroll to the end would undo the jump
    if (reveal.scroll ? reveal.scroll(root) : scrollToTurn(root, reveal.key)) pinnedRef.current = false
  }, [reveal, revealedSeq])

  // Keyboard focus, by key so it stays on its turn when an earlier one goes
  const focusIndex = focusedKey === null ? -1 : turns.findIndex((turn) => turn.key === focusedKey)
  const moveFocus = (step: 1 | -1) => {
    if (turns.length === 0) return
    const from = focusIndex >= 0 ? focusIndex : step > 0 ? -1 : turns.length
    const next = Math.max(0, Math.min(from + step, turns.length - 1))
    // The turns above it are mounted, so it can sit below them
    const above = Math.max(0, next - FOCUS_ROWS_ABOVE)
    if (above < start) setAnchor(above)
    interact()
    pinnedRef.current = false
    setFocusedKey(turns[next].key)
  }
  const toggleFocused = () => {
    const turn = turns[focusIndex]
    if (turn && !turn.live) setToggled((current) => ({ ...current, [turn.key]: !isOpen(turn, focusIndex) }))
    setFocusedKey(null)
  }
  const tabActive = useTabActive()
  // Capture phase so these shortcuts win over editor and input handlers; only from this column's surface (its composer too)
  useEventListener("window", "keydown", (event) => {
    if (isTerminalKeyEvent(event)) return
    const target = event.target
    const surface = scrollRef.current?.parentElement
    if (!(target === document.body || (target instanceof Node && surface?.contains(target)))) return
    if (isModKey(event) && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
      event.preventDefault()
      event.stopPropagation()
      moveFocus(event.key === "ArrowDown" ? 1 : -1)
      scrollRef.current?.focus({ preventScroll: true })
    } else if (event.key === "Enter" && focusIndex >= 0) {
      event.preventDefault()
      event.stopPropagation()
      toggleFocused()
    }
  }, { capture: true, active: tabActive })
  useEscapeLayer(() => setFocusedKey(null), focusIndex >= 0)

  // The focused turn sits FOCUS_ROWS_ABOVE turns below the top, or the first turn at the top
  const scrollToFocus = useEffectEvent(() => {
    const root = scrollRef.current
    if (!root || focusIndex < 0) return
    const atTop = focusIndex < FOCUS_ROWS_ABOVE
    const above = turns[atTop ? 0 : focusIndex - FOCUS_ROWS_ABOVE]
    const element = above && turnElement(root, above.key)
    if (!element) return
    const rect = element.getBoundingClientRect()
    glideTo(root, root.scrollTop + (atTop ? rect.top : rect.bottom) - root.getBoundingClientRect().top)
  })
  useEffect(() => {
    if (focusedKey !== null) scrollToFocus()
  }, [focusedKey])

  /** One expand and one collapse per key, so a memoized turn keeps its props. */
  const [handlers] = useState(() => new Map<string, { expand: () => void; collapse: () => void }>())
  const handlersOf = (key: string) => {
    let entry = handlers.get(key)
    if (!entry) {
      entry = {
        expand: () => {
          setToggled((current) => ({ ...current, [key]: true }))
          setFocusedKey(null)
        },
        collapse: () => setToggled((current) => ({ ...current, [key]: false })),
      }
      handlers.set(key, entry)
    }
    return entry
  }

  return (
    <div ref={attachScroller} onScroll={onScroll} tabIndex={-1} className="min-h-0 flex-1 overflow-y-auto text-ink outline-none" style={{ paddingBottom }}>
      {header}
      <div ref={attachColumn} data-chat-column className="mx-auto" style={{ maxWidth: wide ? undefined : columnWidth }}>
        {start > 0 && (
          <button ref={earlierRef} type="button" onClick={showEarlier} className="mx-auto mt-3 block rounded px-2 py-1 text-xs text-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-1">
            Show earlier turns
          </button>
        )}
        {turns.slice(start).map((turn, offset) => {
          const index = start + offset
          const open = isOpen(turn, index)
          const { expand, collapse } = handlersOf(turn.key)
          const after = renderAfter?.(turn, index)
          return (
            <Fragment key={turn.key}>
              <ColumnTurn
                turnKey={turn.key}
                label={turn.label}
                icon={turn.icon}
                open={open}
                focused={turn.key === focusedKey}
                content={open ? renderBody(turn, index, turn.live ? undefined : collapse) : renderPreview(turn, index)}
                onExpand={expand}
                onResizeStart={wide ? undefined : startResize}
              />
              {after}
            </Fragment>
          )
        })}
        {footer}
      </div>
      <WideModeToggle wide={wide} onToggle={toggleWide} />
    </div>
  )
}
