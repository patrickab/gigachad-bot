"use client"

import { useEffect, useId, useRef, useState, type ElementType, type KeyboardEvent as ReactKeyboardEvent, type ReactNode, type RefObject } from "react"
import { ChevronRight, Ellipsis, Plus, Trash2 } from "lucide-react"
import { cn, errorMessage, toggled } from "@/lib/utils"
import { ErrorText } from "@/components/ui/ErrorText"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { MenuPopover } from "@/components/ui/MenuPopover"
import { Skeleton } from "@/components/ui/Skeleton"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useTabActive } from "@/contexts/TabContext"
import type { FileBranchController } from "./useFileTree"
import type { InlineRename } from "./useInlineRename"

/** A second input of a create form, e.g. the branch a new worktree starts from. */
export interface FileTreeCreateChoice {
  label: string
  options: string[]
  initial: string
  /** A window opens first instead of an inline select: Tab cycles the options, Enter picks one and moves on to the name. */
  floating?: boolean
  /** Floating only: picking one of these options opens its own choice next, e.g. the template after "LaTeX". */
  then?: Record<string, FileTreeCreateChoice>
}

type FileTreeCreateAction =
  | {
    label: string
    placeholder: string
    choice?: FileTreeCreateChoice
    /** `subchoice` is the pick of the `then` choice of `choice`, when it had one. */
    onSubmit: (name: string, choice?: string, subchoice?: string) => Promise<void>
  }
  | {
    label: string
    onClick: () => void | Promise<void>
  }

export interface FileTreeItem<T = unknown> {
  id: string
  label: string
  /** Faint text before the label, e.g. a tab's project and category. */
  prefix?: string
  /** Screen-reader description of the row, e.g. a tab's latest activity. */
  description?: string
  type: "root" | "folder" | "element"
  data?: T
  children?: FileTreeItem<T>[]
  /** Children shown before a "Show more" row, a child holding the active item always shows. */
  childLimit?: number
  isActive?: boolean
  badge?: string | number
  /** Replaces the chevron of an app-generated branch (worktrees, sessions), marking it apart from the files beside it. */
  icon?: ElementType
  /** Space below the row and its contents, separating it from the next sibling. */
  gapAfter?: boolean
  /** Right-aligned row end, e.g. line change counts. The label truncates first. */
  meta?: ReactNode
  draggable?: boolean
  /** What dragging this row carries, matched against `accepts` of drop targets. Default "item". */
  dragKind?: string
  multiSelect?: boolean
  onSelect?: () => void
  /** Double-click renames the row in place (`useInlineRename`). */
  rename?: InlineRename
  onDelete?: () => void
  /** Shown under the row, e.g. an inline confirmation. */
  detail?: ReactNode
  create?: FileTreeCreateAction
  /** The first shows inline on hover (beside `create` only when it is the row's sole other action), the rest go into the "…" menu. */
  actions?: {
    /** Accessible name, e.g. "Archive gigachad-bot"; the inline button's only name. */
    label: string
    /** Short visible text in the "…" menu, e.g. "Archive". Default `label`. */
    menuLabel?: string
    icon: ElementType
    onClick: () => void
    /** Destructive: listed last in the menu, with delete. */
    danger?: boolean
  }[]
  onMoveHere?: (id: string) => Promise<void>
  /** Drag kinds `onMoveHere` takes. Default ["item"]. */
  accepts?: string[]
  onDropFiles?: (paths: string[]) => Promise<void>
}

interface FileTreeProps<T> {
  items: FileTreeItem<T>[]
  controller: FileBranchController
  label: string
  loading?: boolean
  create?: FileTreeCreateAction
  headerActions?: ReactNode
  renderBeforeItem?: (item: FileTreeItem<T>) => ReactNode
  onMoveToRoot?: (id: string) => Promise<void>
  /** Makes the section label a button named `label`, e.g. to open the section's settings. `expanded` makes it a disclosure toggle, `false` hides the items. */
  labelAction?: { label: string; onClick: () => void; expanded?: boolean }
  /** Always-visible content right after the label, e.g. a count and an action. */
  labelSuffix?: ReactNode
}

const FILE_DRAG_TYPE = "application/x-file-tree-ids"
// Drag data is unreadable before the drop, so the kind travels as a data type name, which is readable.
const DRAG_KIND_TYPE_PREFIX = "application/x-file-tree-kind-"
export const DEFAULT_DRAG_KIND = "item"

/** Whether a drag with data `types` carries one of the `accepted` kinds. */
const carriesKind = (types: readonly string[], accepted: string[] = [DEFAULT_DRAG_KIND]) =>
  accepted.some((kind) => types.includes(DRAG_KIND_TYPE_PREFIX + kind))
const INDENT_BASE = 8
const INDENT_STEP = 16
// An open "…" menu inside keeps the actions shown and lets the menu overflow the row. Touch screens, which cannot hover, always show them.
const ACTIONS_CLASS = "flex shrink-0 items-center gap-0.5 max-w-0 overflow-hidden opacity-0 transition-opacity group-hover/row:max-w-32 group-hover/row:opacity-100 group-focus-within/row:max-w-32 group-focus-within/row:opacity-100 has-[[aria-expanded=true]]:max-w-32 has-[[aria-expanded=true]]:overflow-visible has-[[aria-expanded=true]]:opacity-100 [@media(hover:none)]:max-w-32 [@media(hover:none)]:opacity-100"
const ACTION_CLASS = "shrink-0 rounded p-1 text-ink-faint hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
const MENU_ITEM_CLASS = "flex h-7 w-full items-center gap-2 rounded-md px-2 text-left text-xs text-ink-muted transition-colors hover:bg-hover hover:text-ink focus:bg-hover focus:text-ink focus:outline-none"

/** Rows under a nested branch (depth 2+, e.g. a project's files) sit tighter, so a group reads as one. */
function rowHeight(depth: number): string {
  return depth >= 2 ? "h-6" : "h-7"
}

type RowAction = NonNullable<FileTreeItem["actions"]>[number]

/**
 * A row's hover actions: its first action inline, the rest in the "…" menu. Delete (`onDelete`) stays its own inline button.
 *
 * - Beside `create` the first action stays inline only when the menu would hold nothing else
 * - Destructive actions never show inline
 */
function splitActions<T>(item: FileTreeItem<T>): { inline?: RowAction; menu: RowAction[] } {
  const actions = item.actions ?? []
  const inline = actions[0] && !actions[0].danger && (!item.create || actions.length === 1) ? actions[0] : undefined
  return { inline, menu: inline ? actions.slice(1) : actions }
}

/** Children `item` renders: up to `childLimit` unless `uncapped`, plus any later child holding the active item. */
function visibleChildren<T>(item: FileTreeItem<T>, uncapped: boolean): FileTreeItem<T>[] {
  const children = item.children ?? []
  if (uncapped || item.childLimit == null) return children
  const limit = item.childLimit
  return children.filter((child, index) => index < limit || holdsActive(child))
}

function holdsActive<T>(item: FileTreeItem<T>): boolean {
  return Boolean(item.isActive) || (item.children ?? []).some(holdsActive)
}

/** List visible multi-select rows in range-selection order. */
function selectableIds<T>(items: FileTreeItem<T>[], controller: FileBranchController, uncappedIds: ReadonlySet<string>): string[] {
  return items.flatMap((item) => [
    ...(item.multiSelect ? [item.id] : []),
    ...(item.children && controller.isExpanded(item.id) ? selectableIds(visibleChildren(item, uncappedIds.has(item.id)), controller, uncappedIds) : []),
  ])
}

/** Render every hierarchy level with the same indentation, disclosure, selection and action rules. */
export function FileTree<T>({ items, controller, label, loading, create, headerActions, renderBeforeItem, onMoveToRoot, labelAction, labelSuffix }: FileTreeProps<T>) {
  const [creation, setCreation] = useState<{
    parentId: string | null
    action: Extract<FileTreeCreateAction, { onSubmit: unknown }>
  } | null>(null)
  const [actionError, setActionError] = useState("")
  const [renameError, setRenameError] = useState<{ id: string; message: string } | null>(null)
  const [dragOverId, setDragOverId] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set())
  const selectionAnchorId = useRef<string | null>(null)
  const [uncappedIds, setUncappedIds] = useState<ReadonlySet<string>>(() => new Set())
  const descriptionId = useId()

  const toggleCap = (id: string) => setUncappedIds((current) => toggled(current, id))

  const startCreate = async (parentId: string | null, action: FileTreeCreateAction) => {
    setActionError("")
    if (parentId) controller.expand(parentId)
    if ("onSubmit" in action) {
      setCreation({ parentId, action })
      return
    }
    try { await action.onClick() }
    catch (error) { setActionError(errorMessage(error, "Could not create this item")) }
  }

  const handleSelect = (item: FileTreeItem<T>, event: React.MouseEvent) => {
    if (item.multiSelect) {
      if (event.ctrlKey || event.metaKey) {
        setSelectedIds(toggled(selectedIds, item.id))
        selectionAnchorId.current = item.id
        return
      }
      if (event.shiftKey) {
        const visible = selectableIds(items, controller, uncappedIds)
        const clickedIndex = visible.indexOf(item.id)
        const anchorIndex = visible.indexOf(selectionAnchorId.current ?? item.id)
        const from = anchorIndex < 0 ? clickedIndex : Math.min(anchorIndex, clickedIndex)
        const to = anchorIndex < 0 ? clickedIndex : Math.max(anchorIndex, clickedIndex)
        setSelectedIds(new Set(visible.slice(from, to + 1)))
        return
      }
      setSelectedIds(new Set([item.id]))
      selectionAnchorId.current = item.id
    }
    if (item.onSelect) {
      if (item.type !== "element") controller.expand(item.id)
      item.onSelect()
    } else if (item.type !== "element") {
      controller.toggleBranch(item.id)
    }
  }

  const handleDragStart = (event: React.DragEvent, item: FileTreeItem<T>) => {
    if (item.multiSelect) {
      const ids = selectedIds.has(item.id) ? [...selectedIds] : [item.id]
      if (!selectedIds.has(item.id)) setSelectedIds(new Set(ids))
      event.dataTransfer.setData(FILE_DRAG_TYPE, JSON.stringify(ids))
      event.dataTransfer.effectAllowed = "copyMove"
    } else {
      event.dataTransfer.setData("text/plain", item.id)
      event.dataTransfer.setData(DRAG_KIND_TYPE_PREFIX + (item.dragKind ?? DEFAULT_DRAG_KIND), "")
      event.dataTransfer.effectAllowed = "move"
    }
  }

  const handleDrop = async (event: React.DragEvent, item?: FileTreeItem<T>) => {
    event.preventDefault()
    event.stopPropagation()
    setDragOverId(null)
    setActionError("")
    try {
      const files = event.dataTransfer.getData(FILE_DRAG_TYPE)
      if (files) {
        await item?.onDropFiles?.(JSON.parse(files) as string[])
      } else {
        const id = event.dataTransfer.getData("text/plain")
        if (id) await (item ? item.onMoveHere?.(id) : onMoveToRoot?.(id))
      }
    } catch (error) {
      setActionError(errorMessage(error, "Could not move this item"))
    }
  }

  const renderItems = (nodes: FileTreeItem<T>[], depth: number): ReactNode => nodes.map((item) => {
    const branch = item.type !== "element"
    const expanded = branch && controller.isExpanded(item.id)
    // A lone selection is just the last click, so only a Ctrl/Shift multi-selection is shown.
    const selected = selectedIds.size > 1 && selectedIds.has(item.id)
    const paddingLeft = INDENT_BASE + depth * INDENT_STEP
    // A row's icon replaces its chevron, brightening while open.
    const disclosure = item.icon
      ? <item.icon className={cn("h-3 w-3 shrink-0 transition-colors", expanded ? "text-ink" : "text-ink-faint")} />
      : <ChevronRight className={cn("h-3 w-3 shrink-0 transition-transform", expanded && "rotate-90")} />
    const { inline, menu } = splitActions(item)
    return <div key={item.id} className={cn(item.gapAfter && "mb-2")}>
      {!branch && renderBeforeItem && <div style={{ paddingLeft: paddingLeft + 20 }}>
        {renderBeforeItem(item)}
      </div>}
      <div
        className={cn("group/row flex min-w-0 items-center gap-2 rounded-md pr-1 transition-colors duration-160 motion-reduce:transition-none", rowHeight(depth),
          dragOverId === item.id && "ring-1 ring-ink-muted bg-surface-elevated",
          item.multiSelect && "select-none")}
        style={{ marginLeft: paddingLeft }}
        onDragOver={(event) => {
          event.stopPropagation()
          const files = event.dataTransfer.types.includes(FILE_DRAG_TYPE)
          if (files ? !item.onDropFiles : !item.onMoveHere || !carriesKind(event.dataTransfer.types, item.accepts)) return
          event.preventDefault()
          event.dataTransfer.dropEffect = files ? "copy" : "move"
          setDragOverId(item.id)
        }}
        onDragLeave={() => setDragOverId(null)}
        onDrop={(event) => { void handleDrop(event, item) }}
      >
        {branch && item.onSelect && <button
          aria-label={`${expanded ? "Collapse" : "Expand"} ${item.label}`}
          aria-expanded={expanded}
          onClick={() => controller.toggleBranch(item.id)}
          className={cn("flex w-3 shrink-0 items-center text-ink-subtle hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted", rowHeight(depth))}
        >{disclosure}</button>}
        {item.rename?.editing ? <div className={cn("flex min-w-0 flex-1 items-center gap-2", rowHeight(depth))}>
          {!(branch && item.onSelect) && (branch ? disclosure : <span className="w-3 shrink-0" />)}
          <RenameInput label={item.label} rename={item.rename}
            onError={(message) => setRenameError(message ? { id: item.id, message } : null)} />
        </div> : <button
          aria-expanded={branch && !item.onSelect ? expanded : undefined}
          aria-current={!branch && item.isActive ? "page" : undefined}
          aria-describedby={item.description ? `${descriptionId}-${item.id}` : undefined}
          draggable={item.draggable || item.multiSelect}
          onDragStart={(event) => handleDragStart(event, item)}
          onClick={(event) => handleSelect(item, event)}
          onDoubleClick={item.rename?.start}
          className={cn("flex min-w-0 flex-1 items-center gap-2 rounded text-left text-xs focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted", rowHeight(depth),
            item.isActive ? "font-medium text-ink" : item.type === "root" || (branch && depth === 0) ? "text-ink-muted hover:text-ink" : branch ? "text-ink-subtle hover:text-ink" : "text-ink-muted hover:text-ink")}
        >
          {!(branch && item.onSelect) && (branch ? disclosure : <span className="w-3 shrink-0" />)}
          <span
            data-workspace-highlight={item.type === "element" && item.isActive || selected || undefined}
            className={cn("flex min-w-0 items-center gap-2 rounded px-1 transition-colors duration-160 motion-reduce:transition-none",
              (item.type === "element" && item.isActive || selected) && "bg-surface-elevated/70")}
          >
            {item.prefix && <span className="-mr-1 min-w-0 shrink-[2] truncate text-ink-faint">{`${item.prefix} `}</span>}
            <span className="truncate">{item.label}</span>
            {item.badge != null && <span className="shrink-0 text-[10px] text-ink-faint">{item.badge}</span>}
          </span>
          {item.meta != null && <span className="ml-auto flex shrink-0 gap-1.5 text-[10px] tabular-nums">{item.meta}</span>}
        </button>}
        {item.description && <span id={`${descriptionId}-${item.id}`} className="sr-only">{item.description}</span>}
        {!item.rename?.editing && <>
          {(inline || menu.length > 0 || item.onDelete) && <div className={ACTIONS_CLASS}>
            {inline && <button aria-label={inline.label} onClick={inline.onClick} className={ACTION_CLASS}>
              <inline.icon className="h-3 w-3" />
            </button>}
            {menu.length > 0 && <RowMenu label={item.label} actions={menu} />}
            {item.onDelete && <button aria-label={`Delete ${item.label}`} onClick={item.onDelete} className={cn(ACTION_CLASS, "hover:text-danger")}>
              <Trash2 className="h-3 w-3" />
            </button>}
          </div>}
          {/* Creating is the row's main action, so it stays visible at the row end while the rest appears on hover. */}
          {item.create && <button aria-label={item.create.label} onClick={() => { void startCreate(item.id, item.create!) }} className={ACTION_CLASS}>
            <Plus className="h-3 w-3" />
          </button>}
        </>}
      </div>
      {renameError?.id === item.id && item.rename?.editing && <div style={{ marginLeft: paddingLeft + 20 }}>
        <ErrorText className="px-1 py-0.5">{renameError.message}</ErrorText>
      </div>}
      {item.detail && <div style={{ marginLeft: paddingLeft + 20 }}>{item.detail}</div>}
      {expanded && <div role="group" aria-label={`${item.label} contents`} className="motion-expand-in">
        {renderItems(visibleChildren(item, uncappedIds.has(item.id)), depth + 1)}
        {item.children?.length === 0 && creation?.parentId !== item.id && <div
          style={{ marginLeft: INDENT_BASE + (depth + 1) * INDENT_STEP + 20 }} className={cn("flex items-center px-1 text-xs text-ink-faint", rowHeight(depth + 1))}>Empty</div>}
        <CapToggle item={item} depth={depth + 1} uncapped={uncappedIds.has(item.id)} onToggle={() => toggleCap(item.id)} />
        {creation?.parentId === item.id && <InlineCreateForm
          action={creation.action} depth={depth + 1}
          onDone={() => setCreation((current) => current === creation ? null : current)}
          onCancel={() => setCreation(null)}
        />}
      </div>}
    </div>
  })

  const collapsed = labelAction?.expanded === false
  return <div aria-label={label} data-file-tree>
    <div className={cn("group/row flex h-7 items-center justify-between px-2", !collapsed && "mb-1")}>
      <div className="flex items-center gap-1.5">
        {labelAction
          ? <button type="button" aria-label={labelAction.label} aria-expanded={labelAction.expanded} onClick={labelAction.onClick} className="rounded text-xs font-medium text-ink-subtle hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">{label}</button>
          : <span className="text-xs font-medium text-ink-subtle">{label}</span>}
        {labelSuffix}
      </div>
      <div className="flex items-center gap-0.5">
        {headerActions}
        {create && <div className={ACTIONS_CLASS}>
          <button aria-label={create.label} onClick={() => { void startCreate(null, create) }} className={ACTION_CLASS}>
            <Plus className="h-3 w-3" />
          </button>
        </div>}
      </div>
    </div>
    {creation?.parentId === null && <InlineCreateForm action={creation.action} depth={0}
      onDone={() => setCreation((current) => current === creation ? null : current)} onCancel={() => setCreation(null)} />}
    {collapsed ? null : loading ? <div className="space-y-2 px-2 py-1"><Skeleton className="h-3 w-24" /><Skeleton className="h-3 w-32" /></div>
      : <div className="min-h-2"
        onDragOver={(event) => {
          if (event.target !== event.currentTarget || !onMoveToRoot || !carriesKind(event.dataTransfer.types)) return
          event.preventDefault()
          event.dataTransfer.dropEffect = "move"
        }}
        onDrop={(event) => { if (event.target === event.currentTarget) void handleDrop(event) }}
      >{renderItems(items, 0)}</div>}
    <ErrorText className="px-2 py-1">{actionError}</ErrorText>
  </div>
}

/** "Show N more" / "Show less" row under a branch with more children than its `childLimit`. */
function CapToggle<T>({ item, depth, uncapped, onToggle }: {
  item: FileTreeItem<T>
  depth: number
  uncapped: boolean
  onToggle: () => void
}) {
  const total = item.children?.length ?? 0
  if (item.childLimit == null || total <= item.childLimit) return null
  const hidden = total - visibleChildren(item, false).length
  if (!uncapped && hidden === 0) return null
  return <button type="button" onClick={onToggle} style={{ marginLeft: INDENT_BASE + depth * INDENT_STEP + 20 }}
    className="flex h-7 items-center rounded px-1 text-xs text-ink-faint hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
    {uncapped ? "Show less" : `Show ${hidden} more`}
  </button>
}

/** A row's "…" button and its menu of secondary actions, the destructive ones last and set apart. */
function RowMenu({ label, actions }: { label: string; actions: RowAction[] }) {
  const trigger = useRef<HTMLButtonElement>(null)
  return <MenuPopover className="flex" trigger={(open, toggle) => <button ref={trigger} type="button" aria-label={`More actions for ${label}`}
    aria-haspopup="menu" aria-expanded={open} onClick={toggle} className={cn(ACTION_CLASS, open && "text-ink")}>
    <Ellipsis className="h-3 w-3" />
  </button>}>
    {(open, close) => open && <RowMenuList label={label} actions={actions} trigger={trigger} close={close} />}
  </MenuPopover>
}

function RowMenuList({ label, actions, trigger, close }: {
  label: string
  actions: RowAction[]
  trigger: RefObject<HTMLButtonElement | null>
  close: () => void
}) {
  const list = useRef<HTMLDivElement>(null)
  useEffect(() => { list.current?.querySelector<HTMLElement>("[role=menuitem]")?.focus() }, [])
  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    // The menu closes on Escape by itself, focus goes back to its button.
    if (event.key === "Escape") trigger.current?.focus()
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return
    event.preventDefault()
    const items = [...(list.current?.querySelectorAll<HTMLElement>("[role=menuitem]") ?? [])]
    const at = items.indexOf(document.activeElement as HTMLElement)
    items[(at + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length]?.focus()
  }
  const safe = actions.filter((action) => !action.danger)
  const destructive = actions.filter((action) => action.danger)
  const item = (action: RowAction) => <button key={action.label} type="button" role="menuitem"
    aria-label={action.menuLabel && action.label}
    onClick={() => { close(); action.onClick() }}
    className={cn(MENU_ITEM_CLASS, action.danger && "hover:text-danger focus-visible:text-danger")}>
    <action.icon aria-hidden="true" className="h-3 w-3 shrink-0" />
    <span className="truncate">{action.menuLabel ?? action.label}</span>
  </button>
  return <div ref={list} role="menu" aria-label={`${label} actions`} onKeyDown={onKeyDown}
    className="absolute right-0 top-full z-20 mt-1 w-max min-w-36 max-w-56 rounded-lg bg-surface-elevated p-1 shadow-[var(--shadow-md)]">
    {safe.map(item)}
    {safe.length > 0 && destructive.length > 0 && <div role="separator" className="mx-1 my-1 border-t border-divider/50" />}
    {destructive.map(item)}
  </div>
}

function InlineCreateForm({ action, depth, onDone, onCancel }: {
  action: Extract<FileTreeCreateAction, { onSubmit: unknown }>
  depth: number
  onDone: () => void
  onCancel: () => void
}) {
  const [name, setName] = useState("")
  const tabActive = useTabActive()
  useEscapeLayer(onCancel, tabActive)
  const [choice, setChoice] = useState(action.choice?.initial)
  const [subchoice, setSubchoice] = useState<string>()
  /** Open window: "choice" first when floating, then "then" for the `then` choice of the picked option, null once picked. */
  const [picking, setPicking] = useState<"choice" | "then" | null>(action.choice?.floating ? "choice" : null)
  const thenChoice = choice == null ? undefined : action.choice?.then?.[choice]
  const openChoice = picking === "choice" ? action.choice : picking === "then" ? thenChoice : undefined
  const nameInput = useRef<HTMLInputElement>(null)
  const { run, pending: saving, error } = useAsyncAction(async (chosen: string | undefined, subchosen: string | undefined) => {
    await action.onSubmit(name.trim(), chosen, subchosen)
    onDone()
  }, "Could not create this item")
  const submit = () => {
    if (!name.trim() || saving) return
    void run(choice, thenChoice ? subchoice : undefined)
  }
  /** Take the pick of the open window: the first choice opens the `then` choice of its option when there is one. */
  const confirmPick = (chosen: string) => {
    const next = picking === "choice" ? action.choice?.then?.[chosen] : undefined
    if (picking === "choice") {
      setChoice(chosen)
      setSubchoice(next?.initial)
    } else {
      setSubchoice(chosen)
    }
    setPicking(next ? "then" : null)
    // The window unmounts after this commit, then focus returns to the name.
    if (!next) requestAnimationFrame(() => nameInput.current?.focus())
  }
  return <div style={{ marginLeft: INDENT_BASE + depth * INDENT_STEP + 20 }}>
    <input ref={nameInput} autoFocus aria-label={action.placeholder} placeholder={action.placeholder} value={name} disabled={saving}
      onChange={(event) => setName(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault()
          submit()
        }
      }}
      className="h-7 w-full bg-transparent text-xs text-ink placeholder:text-ink-faint outline-none"
    />
    {openChoice && <ChoiceWindow key={picking} choice={openChoice}
      initial={(picking === "choice" ? choice : subchoice) ?? openChoice.initial}
      onCancel={onCancel} onConfirm={confirmPick} />}
    {action.choice?.floating && !picking && <div className="text-[10px] text-ink-faint">{thenChoice && subchoice ? `${choice} · ${subchoice}` : choice}</div>}
    {action.choice && !action.choice.floating && choice != null && <div className="flex items-center gap-1 text-[10px] text-ink-faint">
      <span>{action.choice.label}</span>
      <StyledSelect variant="inline" ariaLabel={action.choice.label} value={choice} onChange={setChoice}
        options={action.choice.options.map((option) => ({ value: option, label: option }))}
        className="text-[10px] text-ink-muted hover:text-ink" />
    </div>}
    <ErrorText className="py-1">{error}</ErrorText>
  </div>
}

function ChoiceWindow({ choice, initial, onConfirm, onCancel }: {
  choice: FileTreeCreateChoice
  initial: string
  onConfirm: (chosen: string) => void
  onCancel: () => void
}) {
  const [selected, setSelected] = useState(initial)
  // Window-level keys: the name field behind the window may hold focus, so the list cannot rely on having it.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const backwards = event.key === "ArrowUp" || (event.key === "Tab" && event.shiftKey)
      const forwards = event.key === "ArrowDown" || (event.key === "Tab" && !event.shiftKey)
      if (event.key !== "Enter" && !backwards && !forwards) return
      event.preventDefault()
      event.stopPropagation()
      if (event.key === "Enter") onConfirm(selected)
      else setSelected(choice.options[(choice.options.indexOf(selected) + (backwards ? -1 : 1) + choice.options.length) % choice.options.length])
    }
    window.addEventListener("keydown", onKeyDown, true)
    return () => window.removeEventListener("keydown", onKeyDown, true)
  }, [choice.options, onConfirm, selected])
  return <FloatingWindow label={choice.label} onClose={onCancel} panelClassName="h-auto w-60 max-w-full rounded-2xl p-2">
    <div role="listbox" aria-label={choice.label} className="flex flex-col">
      {choice.options.map((option) => <div key={option} role="option" aria-selected={option === selected}
        onClick={() => onConfirm(option)}
        className={cn("cursor-pointer rounded-lg px-3 py-1.5 text-sm text-ink-muted", option === selected && "bg-hover text-ink")}>
        {option}
      </div>)}
    </div>
  </FloatingWindow>
}

/** A row's label while it is renamed: Enter saves, leaving saves a changed name, Escape keeps the old one. */
function RenameInput({ label, rename, onError }: { label: string; rename: InlineRename; onError: (message: string) => void }) {
  const [name, setName] = useState(label)
  const [saving, setSaving] = useState(false)
  // Leaving after a refused name drops it instead of asking again
  const refused = useRef<string | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const tabActive = useTabActive()
  const stop = () => {
    onError("")
    rename.stop()
  }
  useEscapeLayer(stop, tabActive)

  useEffect(() => {
    const suffixAt = rename.keepSuffix ? label.lastIndexOf(".") : -1
    input.current?.focus()
    input.current?.setSelectionRange(0, suffixAt > 0 ? suffixAt : label.length)
  }, [label, rename.keepSuffix])

  const save = async (leaving: boolean) => {
    const next = name.trim()
    if (saving) return
    if (!next || next === label || (leaving && next === refused.current)) return stop()
    setSaving(true)
    try {
      await rename.submit(next)
      stop()
    } catch (error) {
      refused.current = next
      onError(errorMessage(error, "Could not rename this item"))
      setSaving(false)
    }
  }

  // Read-only rather than disabled while saving, so a refused name keeps the focus for another try
  return <input ref={input} aria-label={`Rename ${label}`} value={name} readOnly={saving} aria-busy={saving}
    onChange={(event) => setName(event.target.value)}
    onKeyDown={(event) => {
      if (event.key !== "Enter") return
      event.preventDefault()
      void save(false)
    }}
    onBlur={() => { void save(true) }}
    className="h-5 min-w-0 flex-1 rounded bg-surface-elevated px-1 text-xs text-ink outline-none ring-1 ring-ink-muted/40 transition-[box-shadow,opacity] duration-160 focus:ring-ink-muted/70 aria-busy:opacity-60 motion-reduce:transition-none"
  />
}
