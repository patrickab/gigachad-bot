"use client"

import { useId, useRef, useState, type ElementType, type ReactNode } from "react"
import { ChevronRight, Plus, Trash2 } from "lucide-react"
import { cn, errorMessage, toggled } from "@/lib/utils"
import { ErrorText } from "@/components/ui/ErrorText"
import { Skeleton } from "@/components/ui/Skeleton"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useTabActive } from "@/contexts/TabContext"
import type { FileBranchController } from "./useFileTree"

/** A second input of a create form, e.g. the branch a new worktree starts from. */
interface FileTreeCreateChoice {
  label: string
  options: string[]
  initial: string
}

type FileTreeCreateAction =
  | {
    label: string
    placeholder: string
    choice?: FileTreeCreateChoice
    onSubmit: (name: string, choice?: string) => Promise<void>
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
  multiSelect?: boolean
  onSelect?: () => void
  onDoubleClick?: () => void
  onDelete?: () => void
  /** Shown under the row, e.g. an inline confirmation. */
  detail?: ReactNode
  create?: FileTreeCreateAction
  actions?: {
    label: string
    icon: ElementType
    onClick: () => void
  }[]
  onMoveHere?: (id: string) => Promise<void>
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
const INDENT_BASE = 8
const INDENT_STEP = 16
const ACTIONS_CLASS = "flex shrink-0 items-center gap-0.5 max-w-0 overflow-hidden opacity-0 transition-opacity group-hover/row:max-w-32 group-hover/row:opacity-100 group-focus-within/row:max-w-32 group-focus-within/row:opacity-100"
const ACTION_CLASS = "shrink-0 rounded p-1 text-ink-faint hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

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
    return <div key={item.id} className={cn(item.type === "root" && "mt-2 first:mt-0", item.gapAfter && "mb-2")}>
      {!branch && renderBeforeItem && <div style={{ paddingLeft: paddingLeft + 20 }}>
        {renderBeforeItem(item)}
      </div>}
      <div
        className={cn("group/row flex h-6 min-w-0 items-center gap-2 rounded-md pr-1 transition-colors duration-160 motion-reduce:transition-none",
          dragOverId === item.id && "ring-1 ring-ink-muted bg-surface-elevated",
          item.multiSelect && "select-none")}
        style={{ marginLeft: paddingLeft }}
        onDragOver={(event) => {
          event.stopPropagation()
          const files = event.dataTransfer.types.includes(FILE_DRAG_TYPE)
          if (files ? !item.onDropFiles : !item.onMoveHere) return
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
          className="flex h-6 w-3 shrink-0 items-center text-ink-subtle hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
        >{disclosure}</button>}
        <button
          aria-expanded={branch && !item.onSelect ? expanded : undefined}
          aria-current={!branch && item.isActive ? "page" : undefined}
          aria-describedby={item.description ? `${descriptionId}-${item.id}` : undefined}
          draggable={item.draggable || item.multiSelect}
          onDragStart={(event) => handleDragStart(event, item)}
          onClick={(event) => handleSelect(item, event)}
          onDoubleClick={item.onDoubleClick}
          className={cn("flex h-6 min-w-0 flex-1 items-center gap-2 rounded text-left text-xs focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted",
            item.isActive ? "font-medium text-ink" : item.type === "root" ? "font-medium text-ink-muted hover:text-ink" : branch ? "text-ink-subtle hover:text-ink" : "text-ink-muted hover:text-ink")}
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
        </button>
        {item.description && <span id={`${descriptionId}-${item.id}`} className="sr-only">{item.description}</span>}
        <div className={ACTIONS_CLASS}>
          {item.actions?.map((action) => <button key={action.label} aria-label={action.label} onClick={action.onClick} className={ACTION_CLASS}>
            <action.icon className="h-3 w-3" />
          </button>)}
          {item.create && <button aria-label={item.create.label} onClick={() => { void startCreate(item.id, item.create!) }} className={ACTION_CLASS}>
            <Plus className="h-3 w-3" />
          </button>}
          {item.onDelete && <button aria-label={`Delete ${item.label}`} onClick={item.onDelete} className={cn(ACTION_CLASS, "hover:text-danger")}>
            <Trash2 className="h-3 w-3" />
          </button>}
        </div>
      </div>
      {item.detail && <div style={{ marginLeft: paddingLeft + 20 }}>{item.detail}</div>}
      {expanded && <div role="group" aria-label={`${item.label} contents`}>
        {renderItems(visibleChildren(item, uncappedIds.has(item.id)), depth + 1)}
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
    <div className={cn("group/row flex h-6 items-center justify-between px-2", !collapsed && "mb-2")}>
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
          if (event.target !== event.currentTarget || !onMoveToRoot || event.dataTransfer.types.includes(FILE_DRAG_TYPE)) return
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
    className="flex h-6 items-center rounded px-1 text-xs text-ink-faint hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
    {uncapped ? "Show less" : `Show ${hidden} more`}
  </button>
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
  const { run, pending: saving, error } = useAsyncAction(async () => {
    await action.onSubmit(name.trim(), choice)
    onDone()
  }, "Could not create this item")
  const submit = () => {
    if (!name.trim() || saving) return
    void run()
  }
  return <div style={{ marginLeft: INDENT_BASE + depth * INDENT_STEP + 20 }}>
    <input autoFocus aria-label={action.placeholder} placeholder={action.placeholder} value={name} disabled={saving}
      onChange={(event) => setName(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault()
          submit()
        }
      }}
      className="h-6 w-full bg-transparent text-xs text-ink placeholder:text-ink-faint outline-none"
    />
    {action.choice && choice != null && <div className="flex items-center gap-1 text-[10px] text-ink-faint">
      <span>{action.choice.label}</span>
      <StyledSelect variant="inline" ariaLabel={action.choice.label} value={choice} onChange={setChoice}
        options={action.choice.options.map((option) => ({ value: option, label: option }))}
        className="text-[10px] text-ink-muted hover:text-ink" />
    </div>}
    <ErrorText className="py-1">{error}</ErrorText>
  </div>
}
