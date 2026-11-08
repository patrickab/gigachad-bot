"use client"

import { Reorder } from "framer-motion"
import { Check, type LucideIcon } from "lucide-react"
import { useCallback, useEffect, useState, type KeyboardEvent, type ReactNode } from "react"
import { cn } from "@/lib/utils"
import { IconButton } from "./IconButton"
import { PillPopover, usePillPopover } from "./PillPopover"

interface SelectorCategory<T> { key: string; label: string; items: T[] }

interface HeaderAction { icon: LucideIcon; label: string; onClick: () => void }

/** Popover selector: pill trigger, optional category tabs, vertical item list, optional footer.
 *
 * ArrowDown/ArrowUp move focus through the shown items (from outside the list to the selected one, else the first or last), Enter or Space picks the focused one. Escape closes, focus back on the trigger.
 *
 * Attributes
 * ----------
 * triggerLabel : string
 *     Accessible name of the trigger, e.g. `Model: Claude Opus 5`.
 * label : string
 *     Accessible name of the panel.
 * categories : SelectorCategory<T>[]
 *     One category -> plain list, several -> horizontal tabs, opening on the tab holding the selection.
 * renderItem : (item: T) => ReactNode
 *     Row content left of the check mark. Wrap text in a `flex-1 truncate` span.
 * onReorderCategories : ((keys: string[]) => void) | undefined
 *     Set -> tabs are draggable and report the new key order.
 * footer : ReactNode | undefined
 *     Extra content below the list, e.g. an error line.
 * headerAction : HeaderAction | undefined
 *     Icon button right of the tabs, e.g. a settings entry. Keeps the header visible with a single category.
 * view : ReactNode | undefined
 *     Replaces tabs, list and footer while set, e.g. a settings screen opened from the header action.
 * onClose : (() => void) | undefined
 *     Runs whenever the panel closes: selection, outside click or trigger toggle.
 */
interface PopoverSelectorProps<T> {
  triggerLabel: string
  triggerContent: ReactNode
  triggerClassName?: string
  label: string
  categories: SelectorCategory<T>[]
  itemKey: (item: T) => string
  isSelected: (item: T) => boolean
  renderItem: (item: T) => ReactNode
  onSelect: (item: T) => void
  onReorderCategories?: (keys: string[]) => void
  footer?: ReactNode
  headerAction?: HeaderAction
  view?: ReactNode
  emptyText?: string
  placement?: "above" | "below"
  panelClassName?: string
  onClose?: () => void
}

const tabClass = (active: boolean) => cn(
  "w-full whitespace-nowrap rounded-md px-2 py-1.5 text-xs font-medium transition-colors",
  active ? "bg-surface-elevated text-ink shadow-[var(--shadow-sm)]" : "text-ink-subtle hover:text-ink",
)

export function PopoverSelector<T>({
  triggerLabel, triggerContent, triggerClassName, label, categories, itemKey, isSelected, renderItem, onSelect,
  onReorderCategories, footer, headerAction, view, emptyText = "Nothing to select", placement = "above", panelClassName, onClose,
}: PopoverSelectorProps<T>) {
  const [manualCategory, setManualCategory] = useState<string | null>(null)
  const popover = usePillPopover(useCallback(() => {
    setManualCategory(null)
    onClose?.()
  }, [onClose]))
  const { open, close, panelRef } = popover
  // Long lists scroll, so bring the current pick into view on open.
  useEffect(() => {
    if (open) panelRef.current?.querySelector("[data-selected]")?.scrollIntoView?.({ block: "nearest" })
  }, [open, panelRef])

  const active = categories.find((category) => category.key === manualCategory)
    ?? categories.find((category) => category.items.some(isSelected))
    ?? categories[0]
  const select = (item: T) => { onSelect(item); close() }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!open || view !== undefined || (event.key !== "ArrowDown" && event.key !== "ArrowUp")) return
    const items = [...panelRef.current?.querySelectorAll<HTMLElement>("[role=menuitemradio]") ?? []]
    if (items.length === 0) return
    event.preventDefault()
    const down = event.key === "ArrowDown"
    const focused = items.findIndex((row) => row === document.activeElement)
    const selected = items.findIndex((row) => row.hasAttribute("data-selected"))
    const next = focused >= 0 ? (focused + (down ? 1 : items.length - 1)) % items.length
      : selected >= 0 ? selected : down ? 0 : items.length - 1
    items[next].focus()
  }

  const tabButton = (category: SelectorCategory<T>) => (
    <button type="button" onClick={() => setManualCategory(category.key)} className={tabClass(category.key === active?.key)}>{category.label}</button>
  )
  const tabs = categories.length > 1 && (onReorderCategories
    // Tabs size to their labels and scroll sideways once they outgrow the panel.
    ? <Reorder.Group as="div" axis="x" layoutScroll values={categories.map((category) => category.key)} onReorder={onReorderCategories} className="flex min-w-0 flex-1 overflow-x-auto rounded-lg bg-paper p-0.5 [scrollbar-width:none]">
      {categories.map((category) => <Reorder.Item as="div" key={category.key} value={category.key} className="flex-auto shrink-0 cursor-grab active:cursor-grabbing">
        {tabButton(category)}
      </Reorder.Item>)}
    </Reorder.Group>
    : <div className="flex min-w-0 flex-1 overflow-x-auto rounded-lg bg-paper p-0.5 [scrollbar-width:none]">
      {categories.map((category) => <div key={category.key} className="flex-auto shrink-0">{tabButton(category)}</div>)}
    </div>)

  const list = active && active.items.length > 0
    ? <div role="menu" aria-label={active.label} className="max-h-[50vh] space-y-0.5 overflow-y-auto p-1.5">{active.items.map((item) => {
      const selected = isSelected(item)
      return <button
        key={itemKey(item)}
        type="button"
        role="menuitemradio"
        aria-checked={selected}
        onClick={() => select(item)}
        data-selected={selected || undefined}
        className={cn(
          "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:-outline-offset-1 focus-visible:outline-ink-muted",
          selected ? "bg-surface-elevated text-ink" : "text-ink-muted hover:bg-surface-elevated hover:text-ink",
        )}
      >
        {renderItem(item)}
        {selected && <Check className="ml-auto h-4 w-4 shrink-0" />}
      </button>
    })}</div>
    : <p className="px-3.5 py-3 text-xs text-ink-subtle">{emptyText}</p>

  return <PillPopover
    popover={popover}
    triggerLabel={triggerLabel}
    triggerContent={triggerContent}
    triggerClassName={triggerClassName}
    label={label}
    placement={placement}
    panelClassName={cn(view === undefined && "overflow-hidden", panelClassName)}
    onKeyDown={onKeyDown}
  >
    {view ?? <>
      {(tabs || headerAction) && <div className="flex items-center gap-1 border-b border-divider p-1.5">{tabs}{headerAction && <>{!tabs && <span className="flex-1" />}<IconButton {...headerAction} className="hover:bg-surface-elevated" /></>}</div>}
      {list}
      {footer && <div className="border-t border-divider p-1.5">{footer}</div>}
    </>}
  </PillPopover>
}
