"use client"

import { AnimatePresence, motion, Reorder } from "framer-motion"
import { Check, type LucideIcon } from "lucide-react"
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { useClickOutside } from "@/hooks/useClickOutside"
import { cn } from "@/lib/utils"

interface SelectorCategory<T> { key: string; label: string; items: T[] }

/** Popover selector: pill trigger, optional category tabs, vertical item list, optional footer.
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
 *     Extra options below the list, usually `PopoverSelectorAction` rows.
 * view : ReactNode | undefined
 *     Replaces tabs, list and footer while set, e.g. a settings screen opened from the footer.
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

/** Footer row template for extra options such as "Manage models". */
export function PopoverSelectorAction({ icon: Icon, label, onClick }: { icon: LucideIcon; label: string; onClick: () => void }) {
  return <button type="button" onClick={onClick} className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-ink-muted transition-colors hover:bg-surface-elevated hover:text-ink">
    <Icon className="h-3.5 w-3.5" aria-hidden="true" />
    {label}
  </button>
}

export function PopoverSelector<T>({
  triggerLabel, triggerContent, triggerClassName, label, categories, itemKey, isSelected, renderItem, onSelect,
  onReorderCategories, footer, view, emptyText = "Nothing to select", placement = "above", panelClassName, onClose,
}: PopoverSelectorProps<T>) {
  const [open, setOpen] = useState(false)
  const [manualCategory, setManualCategory] = useState<string | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const close = useCallback(() => {
    setOpen(false)
    setManualCategory(null)
    onClose?.()
  }, [onClose])
  useClickOutside(rootRef, close)
  // Long lists scroll, so bring the current pick into view on open.
  useEffect(() => {
    if (open) panelRef.current?.querySelector("[data-selected]")?.scrollIntoView?.({ block: "nearest" })
  }, [open])

  const active = categories.find((category) => category.key === manualCategory)
    ?? categories.find((category) => category.items.some(isSelected))
    ?? categories[0]
  const select = (item: T) => { onSelect(item); close() }

  const tabs = categories.length > 1 && (onReorderCategories
    // Tabs size to their labels and scroll sideways once they outgrow the panel.
    ? <Reorder.Group as="div" axis="x" layoutScroll values={categories.map((category) => category.key)} onReorder={onReorderCategories} className="flex overflow-x-auto rounded-lg bg-paper p-0.5 [scrollbar-width:none]">
      {categories.map((category) => <Reorder.Item as="div" key={category.key} value={category.key} className="flex-auto shrink-0 cursor-grab active:cursor-grabbing">
        <button type="button" onClick={() => setManualCategory(category.key)} className={tabClass(category.key === active?.key)}>{category.label}</button>
      </Reorder.Item>)}
    </Reorder.Group>
    : <div className="flex overflow-x-auto rounded-lg bg-paper p-0.5 [scrollbar-width:none]">
      {categories.map((category) => <div key={category.key} className="flex-auto shrink-0">
        <button type="button" onClick={() => setManualCategory(category.key)} className={tabClass(category.key === active?.key)}>{category.label}</button>
      </div>)}
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
        className={cn("flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors", selected ? "bg-surface-elevated text-ink" : "text-ink-muted hover:bg-surface-elevated hover:text-ink")}
      >
        {renderItem(item)}
        {selected && <Check className="ml-auto h-4 w-4 shrink-0" />}
      </button>
    })}</div>
    : <p className="px-3.5 py-3 text-xs text-ink-subtle">{emptyText}</p>

  return <div ref={rootRef} className="relative min-w-0">
    <button
      type="button"
      onClick={() => open ? close() : setOpen(true)}
      aria-label={triggerLabel}
      aria-haspopup="dialog"
      aria-expanded={open}
      className={cn("flex h-7 min-w-0 items-center gap-1.5 rounded-full px-3 text-xs text-ink-muted transition-colors hover:bg-hover hover:text-ink", open && "text-ink", triggerClassName)}
    >
      {triggerContent}
    </button>
    <AnimatePresence>{open && <motion.div
      ref={panelRef}
      role="dialog"
      aria-label={label}
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 4 }}
      transition={{ duration: 0.15 }}
      className={cn(
        "absolute left-0 z-50 flex w-64 flex-col rounded-xl border border-divider bg-surface shadow-[var(--shadow-lg)]",
        view === undefined && "overflow-hidden",
        placement === "below" ? "top-full mt-2" : "bottom-full mb-2",
        panelClassName,
      )}
    >
      {view ?? <>
        {tabs && <div className="border-b border-divider p-1.5">{tabs}</div>}
        {list}
        {footer && <div className="border-t border-divider p-1.5">{footer}</div>}
      </>}
    </motion.div>}</AnimatePresence>
  </div>
}
