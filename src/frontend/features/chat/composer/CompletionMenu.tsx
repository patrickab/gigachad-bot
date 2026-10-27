import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type RefObject } from "react"
import type { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { useTabActive } from "@/contexts/TabContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"

/** One suggestion for the draft. Applying it replaces `text.slice(from, to)` with `insert`, caret at `from + caretOffset`. */
export interface ComposerCompletion {
  label: string
  detail?: string
  icon: LucideIcon
  /** Monospace label, for paths. */
  mono?: boolean
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
 */
export function useCompletions(
  complete: CompleteDraft | undefined,
  textareaRef: RefObject<HTMLTextAreaElement | null>,
  setText: (text: string) => void,
) {
  const [items, setItems] = useState<ComposerCompletion[]>([])
  const tabActive = useTabActive()
  const [activeIndex, setActiveIndex] = useState(0)
  const latestRequestRef = useRef(0)
  const timerRef = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(timerRef.current), [])

  const close = useCallback(() => {
    latestRequestRef.current += 1
    window.clearTimeout(timerRef.current)
    setItems([])
  }, [])

  const request = useCallback((text: string, caret: number) => {
    if (!complete) return
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

  useEscapeLayer(close, items.length > 0 && tabActive)

  return { items, activeIndex, setActiveIndex, request, close, apply, handleKey }
}

/** Suggestion list above the composer, keyboard state from `useCompletions`. */
export function CompletionMenu({ items, activeIndex, onHover, onApply }: {
  items: ComposerCompletion[]
  activeIndex: number
  onHover: (index: number) => void
  onApply: (item: ComposerCompletion) => void
}) {
  const listRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${activeIndex}"]`)?.scrollIntoView?.({ block: "nearest" })
  }, [activeIndex, items])

  if (items.length === 0) return null
  return (
    <div ref={listRef} role="listbox" aria-label="Suggestions"
      className="absolute inset-x-0 bottom-full mb-2 max-h-72 overflow-y-auto rounded-xl border border-divider bg-paper p-1.5 shadow-[var(--shadow-xl)]"
    >
      {items.map((item, index) => (
        <button key={`${item.from}:${item.label}`} type="button" role="option" data-index={index}
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
          <item.icon className="h-3.5 w-3.5 shrink-0 text-ink-muted" />
          <span className={cn("max-w-[60%] shrink-0 truncate", item.mono && "font-mono")}>{item.label}</span>
          {item.detail && <span className="min-w-0 flex-1 truncate text-ink-subtle">{item.detail}</span>}
        </button>
      ))}
    </div>
  )
}
