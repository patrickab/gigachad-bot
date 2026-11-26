"use client"

import { Fragment, isValidElement, memo, useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from "react"
import { TurnCard, TurnHeader } from "@gigachad/chat-ui"
import "@gigachad/chat-ui/turn.css"
import { ResizeEdges, useColumnWidth, type ResizeStart } from "@/components/ui/columnResize"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { WideModeToggle } from "@/features/chat/WideModeToggle"

/** Turns mounted at first and added per "Show earlier turns": older ones create no elements (smooth streaming). */
export const TURN_PAGE_SIZE = 20
/** Distance from the end (px) that still counts as reading the end, which the view then follows. */
const PINNED_SLACK_PX = 48
/** Load the earlier page while the button is still this far below the top edge, so an upward scroll continues into it. */
const LOAD_EARLIER_ROOT_MARGIN = "400px 0px 0px 0px"

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
  /** Keeps the last turn clear of a floating composer. */
  paddingBottom?: number
  /** The turn column (`[data-chat-column]`), e.g. for the top bar's label to make way for it. */
  onColumnChange?: (column: HTMLDivElement | null) => void
  /** The scroll container, for a caller that paints or listens in it. */
  scrollRef?: RefObject<HTMLDivElement | null>
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

interface ColumnTurnProps {
  turnKey: string
  label: string
  icon: ReactNode
  open: boolean
  /** The preview when collapsed, the card content when open. */
  content: ReactNode
  onExpand: () => void
  onResizeStart?: ResizeStart
}

/** One turn's card; skipped while its props hold, so a streamed frame re-renders the live turn only. */
const ColumnTurn = memo(function ColumnTurn({ turnKey, label, icon, open, content, onExpand, onResizeStart }: ColumnTurnProps) {
  if (!open) {
    return (
      <div data-turn-key={turnKey}>
        <TurnCard expanded={false}>
          <TurnHeader label={label} icon={icon} onExpand={onExpand} />
          {content}
        </TurnCard>
      </div>
    )
  }
  return (
    <div data-turn-key={turnKey} className="relative">
      <ResizeEdges onResizeStart={onResizeStart} />
      <TurnCard expanded>{content}</TurnCard>
    </div>
  )
}, (prev, next) => prev.turnKey === next.turnKey && prev.label === next.label && prev.open === next.open && prev.onExpand === next.onExpand
  && prev.onResizeStart === next.onResizeStart && sameNode(prev.icon, next.icon) && sameNode(prev.content, next.content))

/** Scrolls the turn `key` to the top of the view, true when it is mounted. */
function scrollToTurn(root: HTMLElement, key: string): boolean {
  const element = [...root.querySelectorAll("[data-turn-key]")].find((candidate) => candidate.getAttribute("data-turn-key") === key)
  element?.scrollIntoView({ block: "start", behavior: "instant" })
  return element !== undefined
}

/**
 * The column of a conversation's turns, shared by the chat and Pi: every turn a card, open or collapsed to its header over a preview.
 *
 * - The newest turn is open, a live one always; others open and collapse on click, kept by `turn.key`
 * - Follows the end while the reader is there, `paddingBottom` keeps the last turn clear of the floating composer
 * - Mounts the newest `TURN_PAGE_SIZE` turns, earlier pages on scrolling up (or "Show earlier turns"); while the reader is away
 *   from the end the window's top stays put, so appended turns do not push the read ones out
 * - `reveal` mounts and opens its turn, then scrolls to it (or where its `scroll` says) and holds the view there
 * - Column width (drag a card edge, shared app-wide) and wide mode (the top bar's toggle)
 */
export function ConversationColumn<T extends ConversationTurn>({ turns, renderPreview, renderBody, renderAfter, header, footer, reveal = null, paddingBottom, onColumnChange, scrollRef: outerScrollRef }: ConversationColumnProps<T>) {
  const [toggled, setToggled] = useState<Record<string, boolean>>({})
  const [pageCount, setPageCount] = useState(TURN_PAGE_SIZE)
  /** First mounted turn while the reader is away from the end, null while following the end. */
  const [anchor, setAnchor] = useState<number | null>(null)
  const [wide, setWide] = useState(false)
  const { width: columnWidth, startResize } = useColumnWidth()
  const scrollRef = useRef<HTMLDivElement>(null)
  /** The scroller into `scrollRef` and the caller's. */
  const attachScroller = useCallback((element: HTMLDivElement | null) => {
    scrollRef.current = element
    if (outerScrollRef) outerScrollRef.current = element
  }, [outerScrollRef])
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

  // A reveal mounts and opens its turn while rendering, then the effect scrolls to it, once per `seq`
  useResetOnChange(reveal?.seq ?? null, () => {
    const index = reveal ? turns.findIndex((turn) => turn.key === reveal.key) : -1
    if (!reveal || index < 0) return
    if (index < start) setAnchor(index)
    setToggled((current) => (current[reveal.key] ? current : { ...current, [reveal.key]: true }))
  })
  const revealed = useRef<number | null>(null)
  useEffect(() => {
    const root = scrollRef.current
    if (!reveal || !root || revealed.current === reveal.seq) return
    revealed.current = reveal.seq
    // Held where it lands: a scroll to the end would undo the jump
    if (reveal.scroll ? reveal.scroll(root) : scrollToTurn(root, reveal.key)) pinnedRef.current = false
  }, [reveal])

  /** One expand and one collapse per key, so a memoized turn keeps its props. */
  const [handlers] = useState(() => new Map<string, { expand: () => void; collapse: () => void }>())
  const handlersOf = (key: string) => {
    let entry = handlers.get(key)
    if (!entry) {
      entry = {
        expand: () => setToggled((current) => ({ ...current, [key]: true })),
        collapse: () => setToggled((current) => ({ ...current, [key]: false })),
      }
      handlers.set(key, entry)
    }
    return entry
  }

  return (
    <div ref={attachScroller} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto" style={{ paddingBottom }}>
      {header}
      <div ref={onColumnChange} data-chat-column className="mx-auto" style={{ maxWidth: wide ? undefined : columnWidth }}>
        {start > 0 && (
          <button ref={earlierRef} type="button" onClick={showEarlier} className="mx-auto mt-3 block rounded px-2 py-1 text-xs text-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-1">
            Show earlier turns
          </button>
        )}
        {turns.slice(start).map((turn, offset) => {
          const index = start + offset
          const open = turn.live || (toggled[turn.key] ?? index === turns.length - 1)
          const { expand, collapse } = handlersOf(turn.key)
          const after = renderAfter?.(turn, index)
          return (
            <Fragment key={turn.key}>
              <ColumnTurn
                turnKey={turn.key}
                label={turn.label}
                icon={turn.icon}
                open={open}
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
      <WideModeToggle wide={wide} onToggle={() => setWide((current) => !current)} />
    </div>
  )
}
