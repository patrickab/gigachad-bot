import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type RefObject } from "react"
import type { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useResetOnChange } from "@/hooks/useResetOnChange"

/** One row of the menu above the composer. */
export interface MenuItem {
  label: string
  detail?: string
  icon?: LucideIcon
  /** Monospace label, for paths. */
  mono?: boolean
}

/** One suggestion for the draft. Applying it replaces `text.slice(from, to)` with `insert`, caret at `from + caretOffset`. */
export interface ComposerCompletion extends MenuItem {
  from: number
  to: number
  insert: string
  caretOffset: number
}

/** Suggestions for `text` with the caret at `caret`, empty for none. */
export type CompleteDraft = (text: string, caret: number) => Promise<ComposerCompletion[]>

const REQUEST_DELAY_MS = 80

/**
 * Suggestion state for a composer textarea.
 *
 * - `request` after every edit, debounced, only the latest answer shows
 * - `handleKey` drives the open menu (arrows, Enter/Tab apply) and reports whether it took the key; Escape closes it through an escape layer
 * - Applying asks again, so a directory keeps the menu open for its contents
 * - A new `complete` (its source loaded more) answers the open query again
 */
export function useCompletions(
  complete: CompleteDraft | undefined,
  textareaRef: RefObject<HTMLTextAreaElement | null>,
  setText: (text: string) => void,
) {
  const [items, setItems] = useState<ComposerCompletion[]>([])
  const [activeIndex, setActiveIndex] = useState(0)
  const latestRequestRef = useRef(0)
  const timerRef = useRef<number | undefined>(undefined)
  /** Draft of the latest request, null once closed. */
  const openQueryRef = useRef<{ text: string; caret: number } | null>(null)

  useEffect(() => () => window.clearTimeout(timerRef.current), [])

  const close = useCallback(() => {
    latestRequestRef.current += 1
    openQueryRef.current = null
    window.clearTimeout(timerRef.current)
    setItems([])
  }, [])

  const request = useCallback((text: string, caret: number) => {
    if (!complete) return
    openQueryRef.current = { text, caret }
    const requestId = ++latestRequestRef.current
    window.clearTimeout(timerRef.current)
    timerRef.current = window.setTimeout(() => {
      complete(text, caret).then((next) => {
        if (requestId !== latestRequestRef.current) return
        setItems(next)
        setActiveIndex(0)
      }, () => {})
    }, REQUEST_DELAY_MS)
  }, [complete])

  useEffect(() => {
    const query = openQueryRef.current
    if (query) request(query.text, query.caret)
  }, [request])

  const apply = useCallback((text: string, item: ComposerCompletion) => {
    const next = text.slice(0, item.from) + item.insert + text.slice(item.to)
    const caret = item.from + item.caretOffset
    setText(next)
    close()
    request(next, caret)
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
      textareaRef.current?.setSelectionRange(caret, caret)
    })
  }, [setText, close, request, textareaRef])

  const handleKey = useCallback((event: KeyboardEvent, text: string): boolean => {
    if (items.length === 0 || event.nativeEvent.isComposing) return false
    const step = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0
    if (step !== 0) setActiveIndex((index) => (index + step + items.length) % items.length)
    else if ((event.key === "Enter" && !event.shiftKey) || event.key === "Tab") apply(text, items[activeIndex])
    else return false
    event.preventDefault()
    return true
  }, [items, activeIndex, apply])

  useEscapeLayer(close, items.length > 0)

  return { items, activeIndex, setActiveIndex, request, close, apply, handleKey }
}

/** A row of a `DraftPicker`, `key` identifies the pick. */
export interface PickerItem extends MenuItem {
  key: string
}

/**
 * A choice the draft filters, shown in the menu instead of suggestions, e.g. an omp extension's `select`.
 *
 * Attributes
 * ----------
 * label : string
 *     Accessible name of the menu.
 * items : (query: string) => PickerItem[]
 *     Rows matching the draft, best first.
 */
export interface DraftPicker {
  label: string
  items: (query: string) => PickerItem[]
  onPick: (key: string) => void
  onCancel: () => void
}

/**
 * Keyboard state of a `DraftPicker` over the draft `text`: arrows move, Enter picks, Escape cancels, `onSettled` follows either.
 *
 * - Takes every Enter while open, so the draft (the picker's query) is never sent
 * - Focuses the draft when the picker opens: the send that triggered it may have disabled, and so blurred, the textarea
 */
export function useDraftPicker(picker: DraftPicker | undefined, text: string, focusDraft: () => void, onSettled: () => void) {
  const items = useMemo(() => picker?.items(text) ?? [], [picker, text])
  const [activeIndex, setActiveIndex] = useState(0)
  useResetOnChange(text, () => setActiveIndex(0))
  useResetOnChange(picker, () => setActiveIndex(0))

  const open = picker !== undefined
  useEffect(() => { if (open) focusDraft() }, [open, focusDraft])

  const pick = (key: string) => {
    picker?.onPick(key)
    onSettled()
  }
  useEscapeLayer(() => {
    picker?.onCancel()
    onSettled()
  }, open)

  const handleKey = (event: KeyboardEvent): boolean => {
    if (!picker || event.nativeEvent.isComposing) return false
    const step = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0
    if (step !== 0) {
      if (items.length > 0) setActiveIndex((index) => (index + step + items.length) % items.length)
    } else if (event.key === "Enter" && !event.shiftKey) {
      const item = items[activeIndex]
      if (item) pick(item.key)
    } else return false
    event.preventDefault()
    return true
  }

  return { items, activeIndex, setActiveIndex, pick, handleKey }
}

/** List above the composer, keyboard state from `useCompletions` or `useDraftPicker`. `emptyText` keeps it open without rows. */
export function CompletionMenu<Item extends MenuItem>({ items, activeIndex, onHover, onApply, label = "Suggestions", emptyText }: {
  items: Item[]
  activeIndex: number
  onHover: (index: number) => void
  onApply: (item: Item) => void
  label?: string
  emptyText?: string
}) {
  const listRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${activeIndex}"]`)?.scrollIntoView?.({ block: "nearest" })
  }, [activeIndex, items])

  if (items.length === 0 && emptyText === undefined) return null
  return (
    <div ref={listRef} role="listbox" aria-label={label}
      className="absolute inset-x-0 bottom-full mb-2 max-h-72 overflow-y-auto rounded-xl border border-divider bg-paper p-1.5 shadow-[var(--shadow-xl)]"
    >
      {items.length === 0 && <div className="px-3 py-1.5 text-xs text-ink-subtle">{emptyText}</div>}
      {items.map((item, index) => (
        <button key={`${index}:${item.label}`} type="button" role="option" data-index={index}
          aria-selected={index === activeIndex}
          // Keep focus in the textarea, its blur closes the menu
          onMouseDown={(event) => event.preventDefault()}
          onMouseEnter={() => onHover(index)}
          onClick={() => onApply(item)}
          className={cn(
            "flex w-full items-center gap-2 rounded-md px-3 py-1.5 text-left text-xs text-ink transition-colors",
            index === activeIndex && "bg-surface-elevated",
          )}
        >
          {item.icon && <item.icon className="h-3.5 w-3.5 shrink-0 text-ink-muted" />}
          <span className={cn("max-w-[60%] shrink-0 truncate", item.mono && "font-mono")}>{item.label}</span>
          {item.detail && <span className="min-w-0 flex-1 truncate text-ink-subtle">{item.detail}</span>}
        </button>
      ))}
    </div>
  )
}
