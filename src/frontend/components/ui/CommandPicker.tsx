"use client"

import { Search } from "lucide-react"
import { useCallback, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { useEventListener } from "@/hooks/useEventListener"
import { cn } from "@/lib/utils"
import { PillPopover, surfaceClass, usePillPopover } from "./PillPopover"

interface CommandGroup<T> { key: string; label: string; items: T[] }

/** Popover command picker: pill trigger, a search field over one list of headed groups, an optional preview card beside the highlighted row.
 *
 * The search field takes focus as the panel opens and keeps it, rows show while every word typed is in `searchText(item)`.
 * One row is always highlighted, the first on open and on every filter change. ArrowDown/ArrowUp move it across groups, wrapping, Home/End jump to the ends, the pointer moves it too, Enter or a click picks it. Escape closes, focus back on the trigger.
 *
 * Attributes
 * ----------
 * triggerLabel : string
 *     Accessible name of the trigger, e.g. `Add slide`.
 * label : string
 *     Accessible name of the panel and the list.
 * groups : CommandGroup<T>[]
 *     Headed sections of the list, in order. Groups without a matching item hide.
 * renderItem : (item: T) => ReactNode
 *     Row content.
 * searchPlaceholder : string
 *     Placeholder and accessible name of the search field.
 * searchText : (item: T) => string
 *     Text the typed words are looked up in.
 * renderPreview : ((item: T) => ReactNode) | undefined
 *     Set -> a card beside the panel, level with the highlighted row, shows the item in 16:9. Returning null shows no card.
 * onPick : (item: T) => boolean
 *     Runs the picked item, returning false keeps the panel open, e.g. when the pick could not be applied.
 */
interface CommandPickerProps<T> {
  triggerLabel: string
  triggerContent: ReactNode
  triggerClassName?: string
  label: string
  groups: CommandGroup<T>[]
  itemKey: (item: T) => string
  renderItem: (item: T) => ReactNode
  searchPlaceholder: string
  searchText: (item: T) => string
  renderPreview?: (item: T) => ReactNode
  onPick: (item: T) => boolean
  placement?: "above" | "below"
  panelClassName?: string
}

/** Space between the panel and the preview card, and the least between the card and the window edge. */
const CARD_GAP = 8

export function CommandPicker<T>({
  triggerLabel, triggerContent, triggerClassName, label, groups, itemKey, renderItem, searchPlaceholder, searchText, renderPreview, onPick,
  placement = "above", panelClassName,
}: CommandPickerProps<T>) {
  const [highlightedKey, setHighlightedKey] = useState<string | null>(null)
  const [query, setQuery] = useState("")
  // The list's height when filtering starts, kept while the panel is open so it does not jump as rows come and go
  const [listHeight, setListHeight] = useState<number | null>(null)
  const listRef = useRef<HTMLElement>(null)
  const cardRef = useRef<HTMLDivElement>(null)
  // Set by a key or filter change that moved the highlight, so only those scroll the list
  const revealHighlight = useRef(false)
  // Scrolling the list under a resting pointer must not move the highlight, only a real pointer move does
  const lastPointer = useRef("")
  const baseId = useId()
  const popover = usePillPopover(useCallback(() => {
    setHighlightedKey(null)
    setQuery("")
    setListHeight(null)
    lastPointer.current = ""
  }, []))
  const { open, close, panelRef } = popover

  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  const matches = (item: T) => {
    const text = searchText(item).toLowerCase()
    return words.every((word) => text.includes(word))
  }
  const shownGroups = groups.map((group) => words.length === 0 ? group : { ...group, items: group.items.filter(matches) })
  const shownItems = shownGroups.flatMap((group) => group.items)
  const shownKeys = shownItems.map(itemKey)
  const rowId = (key: string) => `${baseId}-row-${shownKeys.indexOf(key)}`
  // The row moved to last while it is shown, else the first
  const activeKey = highlightedKey !== null && shownKeys.includes(highlightedKey) ? highlightedKey : shownKeys[0] ?? null
  const activeItem = shownItems.find((item) => itemKey(item) === activeKey)
  const preview = activeItem !== undefined ? renderPreview?.(activeItem) : null
  const showCard = open && preview !== null && preview !== undefined && preview !== false
  const pick = (item: T) => { if (onPick(item)) close() }

  const highlightedRow = useCallback(() => panelRef.current?.querySelector<HTMLElement>("[data-highlighted]"), [panelRef])
  useLayoutEffect(() => {
    if (!revealHighlight.current) return
    revealHighlight.current = false
    highlightedRow()?.scrollIntoView?.({ block: "nearest" })
  }, [activeKey, query, highlightedRow])

  // Beside the panel on the right, else the left, top level with the row and kept inside the window.
  const placeCard = useCallback(() => {
    const card = cardRef.current, row = highlightedRow(), panel = panelRef.current
    if (!card || !row || !panel) return
    const bounds = panel.getBoundingClientRect()
    const right = bounds.right + CARD_GAP
    const left = right + card.offsetWidth <= window.innerWidth - CARD_GAP ? right : Math.max(CARD_GAP, bounds.left - CARD_GAP - card.offsetWidth)
    const top = Math.min(row.getBoundingClientRect().top, window.innerHeight - CARD_GAP - card.offsetHeight)
    card.style.left = `${left}px`
    card.style.top = `${Math.max(CARD_GAP, top)}px`
  }, [highlightedRow, panelRef])
  useLayoutEffect(() => { if (showCard) placeCard() }, [showCard, activeKey, placeCard])
  useEventListener("window", "resize", placeCard, { active: showCard })

  const onSearchKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.nativeEvent.isComposing) return
    if (event.key === "Enter") {
      if (activeItem === undefined) return
      event.preventDefault()
      pick(activeItem)
      return
    }
    const count = shownKeys.length
    const index = activeKey === null ? 0 : shownKeys.indexOf(activeKey)
    const next = event.key === "ArrowDown" ? (index + 1) % count
      : event.key === "ArrowUp" ? (index - 1 + count) % count
      : event.key === "Home" ? 0
      : event.key === "End" ? count - 1
      : null
    if (next === null || count === 0) return
    event.preventDefault()
    if (shownKeys[next] === activeKey) return
    revealHighlight.current = true
    setHighlightedKey(shownKeys[next])
  }

  const onSearchChange = (value: string) => {
    if (listHeight === null) setListHeight(listRef.current?.offsetHeight ?? null)
    setQuery(value)
    setHighlightedKey(null)
    revealHighlight.current = true
  }

  const itemRow = (item: T) => {
    const key = itemKey(item)
    const isHighlighted = key === activeKey
    return <div
      key={key}
      id={rowId(key)}
      role="option"
      aria-selected={isHighlighted}
      onClick={() => pick(item)}
      // The search field keeps focus while the pointer picks
      onMouseDown={(event) => event.preventDefault()}
      onMouseMove={(event) => {
        const at = `${event.clientX},${event.clientY}`
        if (at === lastPointer.current) return
        lastPointer.current = at
        setHighlightedKey(key)
      }}
      data-highlighted={isHighlighted || undefined}
      className={cn(
        "flex w-full cursor-default items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition-colors duration-100",
        isHighlighted ? "bg-surface-elevated text-ink" : "text-ink-muted",
      )}
    >{renderItem(item)}</div>
  }

  const listId = `${baseId}-list`
  const lockedHeight = listHeight !== null ? { minHeight: listHeight } : undefined
  const list = shownItems.length > 0
    ? <div
      ref={(element) => { listRef.current = element }}
      id={listId}
      role="listbox"
      aria-label={label}
      onScroll={placeCard}
      style={lockedHeight}
      className="max-h-[50vh] space-y-0.5 overflow-y-auto p-1.5"
    >{shownGroups.map((group, index) => group.items.length > 0 && <div key={group.key} role="group" aria-labelledby={`${baseId}-${index}`} className="space-y-0.5">
      <p id={`${baseId}-${index}`} className="px-2 pb-0.5 pt-1.5 text-[11px] font-medium text-ink-subtle">{group.label}</p>
      {group.items.map(itemRow)}
    </div>)}</div>
    : <p ref={(element) => { listRef.current = element }} style={lockedHeight} className="px-3.5 py-3 text-xs text-ink-subtle">{words.length > 0 ? "No match" : "Nothing to select"}</p>

  return <>
    <PillPopover
      popover={popover}
      triggerLabel={triggerLabel}
      triggerContent={triggerContent}
      triggerClassName={triggerClassName}
      label={label}
      placement={placement}
      panelClassName={cn("overflow-hidden", panelClassName)}
      onAnimationComplete={placeCard}
    >
      <div className="flex items-center gap-2 border-b border-divider px-3.5">
        <Search className="h-3.5 w-3.5 shrink-0 text-ink-faint" />
        <input
          autoFocus
          role="combobox"
          aria-label={searchPlaceholder}
          aria-expanded={shownItems.length > 0}
          aria-controls={shownItems.length > 0 ? listId : undefined}
          aria-activedescendant={activeKey !== null ? rowId(activeKey) : undefined}
          aria-autocomplete="list"
          placeholder={searchPlaceholder}
          value={query}
          onChange={(event) => onSearchChange(event.target.value)}
          onKeyDown={onSearchKeyDown}
          spellCheck={false}
          autoComplete="off"
          className="h-10 min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-ink-faint"
        />
      </div>
      {list}
    </PillPopover>
    {showCard && createPortal(<div ref={cardRef} aria-hidden className={cn("pointer-events-none fixed z-50 w-72 p-1.5 transition-[top] duration-100 ease-out", surfaceClass)}>
      <div className="aspect-video overflow-hidden rounded-md border border-divider">{preview}</div>
    </div>, document.body)}
  </>
}
