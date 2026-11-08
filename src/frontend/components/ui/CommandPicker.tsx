"use client"

import { Search } from "lucide-react"
import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { useIsPresent } from "framer-motion"
import { useTabActive } from "@/contexts/TabContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useEventListener } from "@/hooks/useEventListener"
import { useOutsidePress } from "@/hooks/useOutsidePress"
import { cn } from "@/lib/utils"
import { FLOATING_PANEL_CLASS, FloatingPanel } from "./FloatingWindow"
import { ModalBackdrop } from "./ModalBackdrop"
import { PillPopover, usePillPopover, type PillPopoverState } from "./PillPopover"

/** Look of the list and the preview window alike, the floating window's panel at a radius that suits a small list. */
const PICKER_PANEL_CLASS = cn(FLOATING_PANEL_CLASS, "rounded-2xl")
/** Corner radius of `PICKER_PANEL_CLASS` (rounded-2xl), a tab previewed in the window is clipped to it. */
export const PICKER_PANEL_RADIUS_PX = 16

/** Headed section of a picker list, `label` omitted for a list without headings. */
export interface PickerGroup<T> { key: string; label?: string; items: T[] }

/**
 * List and preview of every picker: a search field over one list of headed groups, the highlighted row's preview in a floating window.
 *
 * The search field takes focus as the list opens and keeps it. One row is always highlighted, `initialKey` or the first on open,
 * the first on every filter change. ArrowDown/ArrowUp move it across groups, wrapping, Home/End jump to the ends, the pointer
 * moves it too, Enter or a click picks it.
 *
 * Attributes
 * ----------
 * label : string
 *     Accessible name of the list.
 * groups : PickerGroup<T>[]
 *     Sections of the list, in order. Groups without a shown item hide.
 * searchPlaceholder : string
 *     Placeholder and accessible name of the search field.
 * searchText : ((item: T) => string) | undefined
 *     Set -> rows show while every word typed is in it.
 *     Unset -> `groups` are already filtered by the caller, who owns the search through `query` and `onQueryChange`.
 * header : ReactNode | undefined
 *     Row between the search field and the list, e.g. the folder shown.
 * renderPreview : ((item: T) => ReactNode) | undefined
 *     Set -> the page dims while the list is open and the highlighted row shows in a floating window at `previewAspect`.
 *     The list docks over the window's top left corner, fit to its rows. Content that must stay clear of it pads by `var(FINDER_INSET_VAR)`.
 *     Returning null shows no window.
 * previewAspect : number | undefined
 *     Width over height of the preview window, the window's own by default: a tab or file previews as the page it opens.
 * searchAction : ((close: () => void) => ReactNode) | undefined
 *     Set -> a control at the end of the search row, e.g. a create button.
 * onKeyDown : ((event, item: T | undefined, list: PickerKeyContext) => boolean) | undefined
 *     Keys the list does not handle, with the highlighted item and the shown row keys in order. Returning true stops the list's own handling.
 * onPick : (item: T) => boolean
 *     Runs the picked item, returning false keeps the list open, e.g. a folder that opened inside it.
 */
interface PickerListProps<T> {
  label: string
  groups: PickerGroup<T>[]
  itemKey: (item: T) => string
  renderItem: (item: T) => ReactNode
  searchPlaceholder: string
  searchText?: (item: T) => string
  query?: string
  onQueryChange?: (query: string) => void
  header?: ReactNode
  emptyText?: string
  renderPreview?: (item: T) => ReactNode
  previewAspect?: number
  searchAction?: (close: () => void) => ReactNode
  initialKey?: string
  onKeyDown?: (event: KeyboardEvent<HTMLInputElement>, item: T | undefined, list: PickerKeyContext) => boolean
  onPick: (item: T) => boolean
}

/** The shown rows of a picker for a key handler: `keys` in order, `highlight` moves the highlight to one of them. */
export interface PickerKeyContext { keys: string[]; highlight: (key: string) => void }

/** A bare Delete with the search caret at its end (before that it edits the search): the key that removes the highlighted row. */
export function isRemoveRowKey(event: KeyboardEvent<HTMLInputElement>): boolean {
  const input = event.currentTarget
  const caretAtEnd = input.selectionStart === input.value.length && input.selectionEnd === input.value.length
  return event.key === "Delete" && caretAtEnd && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey
}

/** The row after `key` in `list`, the one before it at the end: where the highlight goes once `key`'s row is removed. */
export function neighbourKey(list: PickerKeyContext, key: string): string | undefined {
  const index = list.keys.indexOf(key)
  return list.keys[index + 1] ?? list.keys[index - 1]
}

/** Margin of a picker's preview to the window edges, and the finder's inset into the preview's corner. */
const PICKER_MARGIN = 24
const PICKER_GAP = 12
/** The top bar (h-11) stays clear of the preview. */
const TOP_BAR_PX = 44
/**
 * The finder of a picker with a preview: placed over the preview's corner by `PickerPreview`.
 *
 * - As wide as its rows need, within bounds, the widest row sets it while nothing is typed
 * - A row that should not widen the finder caps its own text, e.g. `truncate max-w-[18rem]`
 */
const FINDER_DOCK_CLASS = "fixed w-max min-w-56 max-w-[min(36rem,calc(100vw-6rem))]"
/** CSS variable on the preview window: the inset from its left edge that clears the finder docked over it. */
export const FINDER_INSET_VAR = "--picker-finder-inset"

export interface Box { left: number; top: number; width: number; height: number }

/**
 * Where a picker with a preview goes: the preview centered on the window, its finder over the preview's top left corner.
 *
 * - The preview is the largest box of `aspect` within symmetric margins that keep the top bar clear, exactly centered
 * - The finder lies inside it by `PICKER_GAP`, the preview gives up its corner rather than its size or center
 */
export function pickerLayout(viewport: { width: number; height: number }, aspect: number): { preview: Box; finder: { left: number; top: number; maxHeight: number } } {
  const vertical = TOP_BAR_PX + PICKER_MARGIN / 2
  const width = Math.max(0, Math.min(viewport.width - 2 * PICKER_MARGIN, (viewport.height - 2 * vertical) * aspect))
  const height = width / aspect
  const preview = { left: (viewport.width - width) / 2, top: (viewport.height - height) / 2, width, height }
  return { preview, finder: { left: preview.left + PICKER_GAP, top: preview.top + PICKER_GAP, maxHeight: height - 2 * PICKER_GAP } }
}

function isInside(box: Box, x: number, y: number): boolean {
  return x >= box.left && x <= box.left + box.width && y >= box.top && y <= box.top + box.height
}

/** `root` or its first descendant that scrolls vertically, in document order. */
function firstScrollable(root: HTMLElement | null): HTMLElement | null {
  if (!root) return null
  const scrolls = (element: HTMLElement) => element.scrollHeight > element.clientHeight && /auto|scroll/.test(getComputedStyle(element).overflowY)
  return [root, ...root.querySelectorAll<HTMLElement>("*")].find(scrolls) ?? null
}

/** The dimmed page and the floating preview window centered on it, the `finder` panel moved over its top left corner, `finderWidth` fixes its width when set. A click on the dimmed page runs `onClose`. */
function PickerPreview({ finder, finderWidth, aspect, onClose, children }: { finder: HTMLElement | null; finderWidth: number | null; aspect: number; onClose: () => void; children: ReactNode }) {
  const [box, setBox] = useState<Box | null>(null)
  const [finderInset, setFinderInset] = useState(0)
  useLayoutEffect(() => {
    const place = () => {
      const layout = pickerLayout({ width: window.innerWidth, height: window.innerHeight }, aspect)
      setBox(layout.preview)
      if (!finder) return
      const { left, top, maxHeight } = layout.finder
      Object.assign(finder.style, { left: `${left}px`, top: `${top}px`, maxHeight: `${maxHeight}px` }, finderWidth === null ? {} : { width: `${finderWidth}px` })
    }
    place()
    window.addEventListener("resize", place)
    return () => window.removeEventListener("resize", place)
  }, [aspect, finder, finderWidth])
  // offsetWidth ignores the finder's opening scale animation
  useLayoutEffect(() => {
    if (!finder) return
    const measure = () => setFinderInset(finder.offsetWidth + 2 * PICKER_GAP)
    measure()
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(measure)
    observer.observe(finder)
    return () => observer.disconnect()
  }, [finder])
  // The window lets clicks through to the dimmed page, so a wheel over it scrolls its first scrollable content instead
  const content = useRef<HTMLDivElement>(null)
  useEventListener("window", "wheel", (event) => {
    if (!box || finder?.contains(event.target as Node) || !isInside(box, event.clientX, event.clientY)) return
    firstScrollable(content.current)?.scrollBy({ top: event.deltaY, left: event.deltaX })
  }, { passive: true })
  const shown = children !== null && children !== undefined && children !== false
  // The page washes out in its own paper color and blurs, a modal's dark scrim would turn a light page grey
  return createPortal(<ModalBackdrop onClose={onClose} className="z-[55] bg-paper/50 backdrop-blur-[3px]">
    {shown && box && <FloatingPanel aria-hidden className={cn(PICKER_PANEL_CLASS, "pointer-events-none fixed")}
      style={{ left: box.left, top: box.top, width: box.width, height: box.height, [FINDER_INSET_VAR]: `${finderInset}px` } as CSSProperties}>
      <div ref={content} className="contents">{children}</div>
    </FloatingPanel>}
  </ModalBackdrop>, document.body)
}

/** The search field, list and preview of an open picker, `close` ends it. Remounted per opening, so each starts unfiltered. */
function PickerList<T>({
  label, groups, itemKey, renderItem, searchPlaceholder, searchText, query: controlledQuery, onQueryChange, header, emptyText = "Nothing to select",
  renderPreview, previewAspect, searchAction, initialKey, onKeyDown, onPick, close,
}: PickerListProps<T> & { close: () => void }) {
  // A closed list fades out in place: unreachable meanwhile, its preview and dimmed page gone at once
  const present = useIsPresent()
  const [highlightedKey, setHighlightedKey] = useState<string | null>(null)
  const [ownQuery, setOwnQuery] = useState("")
  const query = controlledQuery ?? ownQuery
  // The list's height and a preview's finder width when filtering starts, kept while it is open so neither jumps as rows come and go
  const [listHeight, setListHeight] = useState<number | null>(null)
  const [finderWidth, setFinderWidth] = useState<number | null>(null)
  const [root, setRoot] = useState<HTMLDivElement | null>(null)
  const listRef = useRef<HTMLElement>(null)
  // Set by a key or filter change that moved the highlight, so only those scroll the list
  const revealHighlight = useRef(false)
  // Scrolling the list under a resting pointer must not move the highlight, only a real pointer move does
  const lastPointer = useRef("")
  const baseId = useId()

  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  const matches = (item: T) => {
    const text = searchText!(item).toLowerCase()
    return words.every((word) => text.includes(word))
  }
  const shownGroups = searchText && words.length > 0 ? groups.map((group) => ({ ...group, items: group.items.filter(matches) })) : groups
  const shownItems = shownGroups.flatMap((group) => group.items)
  const shownKeys = shownItems.map(itemKey)
  const rowId = (key: string) => `${baseId}-row-${shownKeys.indexOf(key)}`
  // The row moved to last while it is shown, else `initialKey` while nothing is typed, else the first
  const restingKey = initialKey !== undefined && words.length === 0 && shownKeys.includes(initialKey) ? initialKey : shownKeys[0] ?? null
  const activeKey = highlightedKey !== null && shownKeys.includes(highlightedKey) ? highlightedKey : restingKey
  const activeItem = shownItems.find((item) => itemKey(item) === activeKey)
  const pick = (item: T) => { if (onPick(item)) close() }

  useLayoutEffect(() => {
    if (!revealHighlight.current) return
    revealHighlight.current = false
    root?.querySelector<HTMLElement>("[data-highlighted]")?.scrollIntoView?.({ block: "nearest" })
  }, [activeKey, query, root])

  const onSearchKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.nativeEvent.isComposing) return
    if (onKeyDown?.(event, activeItem, { keys: shownKeys, highlight: setHighlightedKey })) {
      revealHighlight.current = true
      return
    }
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
    if (listHeight === null) {
      setListHeight(listRef.current?.offsetHeight ?? null)
      // A preview's finder fits its rows: keep the width it opened with while rows filter away
      setFinderWidth(root?.parentElement?.offsetWidth ?? null)
    }
    if (onQueryChange) onQueryChange(value)
    else setOwnQuery(value)
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
        "group flex w-full cursor-default items-center gap-2 rounded-md px-2 py-1 text-left text-xs transition-colors duration-100",
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
      style={lockedHeight}
      className="max-h-[60vh] min-h-0 overflow-y-auto p-1"
    >{shownGroups.map((group, index) => group.items.length > 0 && <div key={group.key} role="group" aria-labelledby={group.label ? `${baseId}-${index}` : undefined}>
      {group.label && <p id={`${baseId}-${index}`} className="px-2 pb-0.5 pt-1.5 text-[10px] font-medium text-ink-subtle">{group.label}</p>}
      {group.items.map(itemRow)}
    </div>)}</div>
    : <p ref={(element) => { listRef.current = element }} style={lockedHeight} className="px-3 py-3 text-xs text-ink-subtle">{words.length > 0 ? "No match" : emptyText}</p>

  return <div ref={setRoot} inert={!present} aria-hidden={!present || undefined} className="flex min-h-0 flex-col">
    <div className="flex items-center gap-2 border-b border-divider px-3">
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
        className="h-9 min-w-0 flex-1 bg-transparent text-xs text-ink outline-none placeholder:text-ink-faint"
      />
      {searchAction?.(close)}
    </div>
    {header && <div className="flex min-w-0 items-center gap-1.5 border-b border-divider px-3 py-1.5 text-[10px] text-ink-faint">{header}</div>}
    {list}
    {renderPreview && present && <PickerPreview finder={root?.parentElement ?? null} finderWidth={finderWidth} aspect={previewAspect ?? window.innerWidth / window.innerHeight} onClose={close}>
      {activeItem !== undefined ? renderPreview(activeItem) : null}
    </PickerPreview>}
  </div>
}

/** Picker behind a pill trigger: the list opens at the trigger, the preview floats over the page. See `PickerList` for the rest. */
export function CommandPicker<T>({ triggerLabel, triggerContent, triggerClassName, placement = "above", panelClassName, popover: controlledPopover, ...list }: PickerListProps<T> & {
  triggerLabel: string
  triggerContent: ReactNode
  triggerClassName?: string
  placement?: "above" | "below"
  panelClassName?: string
  /** Set when the caller opens the picker itself, e.g. from a shortcut. */
  popover?: PillPopoverState
}) {
  const ownPopover = usePillPopover()
  const popover = controlledPopover ?? ownPopover
  // The list lies in a portal, outside the tab: a tab sent to the background takes its open list along
  const tabActive = useTabActive()
  const { open, close } = popover
  useEffect(() => { if (!tabActive && open) close() }, [tabActive, open, close])
  return <PillPopover
    popover={popover}
    triggerLabel={triggerLabel}
    triggerContent={triggerContent}
    triggerClassName={triggerClassName}
    label={list.label}
    placement={placement}
    portal={list.renderPreview ? "docked" : "trigger"}
    panelClassName={cn(PICKER_PANEL_CLASS, list.renderPreview && FINDER_DOCK_CLASS, panelClassName)}
  >
    {/* PillPopover mounts its panel only while open, its exit animation keeps the last list */}
    <PickerList {...list} close={popover.close} />
  </PillPopover>
}

/** Picker opened without a trigger (a shortcut): the list docks under the top bar's left end, `onClose` ends it. See `PickerList`. */
export function FloatingPicker<T>({ onClose, panelClassName, ...list }: PickerListProps<T> & { onClose: () => void; panelClassName?: string }) {
  const panelRef = useRef<HTMLDivElement>(null)
  // Escape above every layer underneath, a press outside the list (the preview's dimmed page too) closes it
  useEscapeLayer(onClose)
  useOutsidePress(panelRef, onClose)
  return createPortal(<div ref={panelRef} role="dialog" aria-label={list.label}
    className={cn("z-[60] w-80", list.renderPreview ? FINDER_DOCK_CLASS : "fixed left-3 top-12", PICKER_PANEL_CLASS, panelClassName)}>
    <PickerList {...list} close={onClose} />
  </div>, document.body)
}
