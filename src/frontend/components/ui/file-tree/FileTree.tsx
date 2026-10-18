"use client"

import { useRef, useState, type ElementType, type ReactNode } from "react"
import { ChevronRight, Plus, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Skeleton } from "@/components/ui/Skeleton"
import type { FileBranchController } from "./useFileTree"

export type FileTreeCreateAction =
  | {
    label: string
    placeholder: string
    onSubmit: (name: string) => Promise<void>
  }
  | {
    label: string
    onClick: () => void | Promise<void>
  }

export interface FileTreeItem<T = unknown> {
  id: string
  label: string
  type: "root" | "folder" | "element"
  data?: T
  children?: FileTreeItem<T>[]
  isActive?: boolean
  badge?: string | number
  draggable?: boolean
  multiSelect?: boolean
  onSelect?: () => void
  onDelete?: () => void
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
  renderBeforeItem?: (item: FileTreeItem<T>) => ReactNode
  onMoveToRoot?: (id: string) => Promise<void>
}

const FILE_DRAG_TYPE = "application/x-file-tree-ids"
const INDENT_BASE = 8
const INDENT_STEP = 16
const ACTIONS_CLASS = "flex shrink-0 items-center gap-0.5 max-w-0 overflow-hidden opacity-0 transition-opacity group-hover/row:max-w-24 group-hover/row:opacity-100 group-focus-within/row:max-w-24 group-focus-within/row:opacity-100"
const ACTION_CLASS = "shrink-0 rounded p-1 text-ink-faint hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** List visible multi-select rows in range-selection order. */
function selectableIds<T>(items: FileTreeItem<T>[], controller: FileBranchController): string[] {
  return items.flatMap((item) => [
    ...(item.multiSelect ? [item.id] : []),
    ...(item.children && controller.isExpanded(item.id) ? selectableIds(item.children, controller) : []),
  ])
}

/** Render every hierarchy level with the same indentation, disclosure, selection and action rules. */
export function FileTree<T>({ items, controller, label, loading, create, renderBeforeItem, onMoveToRoot }: FileTreeProps<T>) {
  const [creation, setCreation] = useState<{
    parentId: string | null
    action: Extract<FileTreeCreateAction, { onSubmit: unknown }>
  } | null>(null)
  const [actionError, setActionError] = useState("")
  const [dragOverId, setDragOverId] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set())
  const selectionAnchorId = useRef<string | null>(null)

  const startCreate = async (parentId: string | null, action: FileTreeCreateAction) => {
    setActionError("")
    if (parentId) controller.expand(parentId)
    if ("onSubmit" in action) {
      setCreation({ parentId, action })
      return
    }
    try { await action.onClick() }
    catch (error) { setActionError(error instanceof Error ? error.message : "Could not create this item") }
  }

  const handleSelect = (item: FileTreeItem<T>, event: React.MouseEvent) => {
    if (item.multiSelect) {
      if (event.ctrlKey || event.metaKey) {
        const next = new Set(selectedIds)
        if (!next.delete(item.id)) next.add(item.id)
        setSelectedIds(next)
        selectionAnchorId.current = item.id
        return
      }
      if (event.shiftKey) {
        const visible = selectableIds(items, controller)
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
      setActionError(error instanceof Error ? error.message : "Could not move this item")
    }
  }

  const renderItems = (nodes: FileTreeItem<T>[], depth: number): ReactNode => nodes.map((item) => {
    const branch = item.type !== "element"
    const expanded = branch && controller.isExpanded(item.id)
    const selected = selectedIds.has(item.id)
    const paddingLeft = INDENT_BASE + depth * INDENT_STEP
    return <div key={item.id} className={item.type === "root" ? "mt-2 first:mt-0" : undefined}>
      {!branch && renderBeforeItem && <div style={{ paddingLeft: paddingLeft + 20 }}>
        {renderBeforeItem(item)}
      </div>}
      <div
        data-workspace-highlight={item.type === "element" && item.isActive || selected || undefined}
        className={cn("group/row flex h-6 min-w-0 items-center gap-2 rounded-md pr-1 transition-colors duration-160 motion-reduce:transition-none",
          (item.type === "element" && item.isActive || selected) && "bg-surface-elevated/70",
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
        ><ChevronRight className={cn("h-3 w-3 transition-transform", expanded && "rotate-90")} /></button>}
        <button
          aria-expanded={branch && !item.onSelect ? expanded : undefined}
          aria-current={!branch && item.isActive ? "page" : undefined}
          draggable={item.draggable || item.multiSelect}
          onDragStart={(event) => handleDragStart(event, item)}
          onClick={(event) => handleSelect(item, event)}
          className={cn("flex h-6 min-w-0 flex-1 items-center gap-2 rounded text-left text-xs focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted",
            item.isActive ? "font-medium text-ink" : item.type === "root" ? "font-medium text-ink-muted hover:text-ink" : branch ? "text-ink-subtle hover:text-ink" : "text-ink-muted hover:text-ink")}
        >
          {!(branch && item.onSelect) && (branch
            ? <ChevronRight className={cn("h-3 w-3 shrink-0 transition-transform", expanded && "rotate-90")} />
            : <span className="w-3 shrink-0" />)}
          <span className="truncate">{item.label}</span>
          {item.badge != null && <span className="shrink-0 text-[10px] text-ink-faint">{item.badge}</span>}
        </button>
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
      {expanded && <div role="group" aria-label={`${item.label} contents`}>
        {renderItems(item.children ?? [], depth + 1)}
        {creation?.parentId === item.id && <InlineCreateForm
          action={creation.action} depth={depth + 1}
          onDone={() => setCreation((current) => current === creation ? null : current)}
          onCancel={() => setCreation(null)}
        />}
      </div>}
    </div>
  })

  return <div aria-label={label} data-file-tree>
    <div className="group/row mb-2 flex h-6 items-center justify-between px-2">
      <span className="text-xs font-medium text-ink-subtle">{label}</span>
      {create && <div className={ACTIONS_CLASS}>
        <button aria-label={create.label} onClick={() => { void startCreate(null, create) }} className={ACTION_CLASS}>
          <Plus className="h-3 w-3" />
        </button>
      </div>}
    </div>
    {creation?.parentId === null && <InlineCreateForm action={creation.action} depth={0}
      onDone={() => setCreation((current) => current === creation ? null : current)} onCancel={() => setCreation(null)} />}
    {loading ? <div className="space-y-2 px-2 py-1"><Skeleton className="h-3 w-24" /><Skeleton className="h-3 w-32" /></div>
      : <div className="min-h-2"
        onDragOver={(event) => {
          if (event.target !== event.currentTarget || !onMoveToRoot || event.dataTransfer.types.includes(FILE_DRAG_TYPE)) return
          event.preventDefault()
          event.dataTransfer.dropEffect = "move"
        }}
        onDrop={(event) => { if (event.target === event.currentTarget) void handleDrop(event) }}
      >{renderItems(items, 0)}</div>}
    {actionError && <p role="alert" className="px-2 py-1 text-xs text-danger">{actionError}</p>}
  </div>
}

function InlineCreateForm({ action, depth, onDone, onCancel }: {
  action: Extract<FileTreeCreateAction, { onSubmit: unknown }>
  depth: number
  onDone: () => void
  onCancel: () => void
}) {
  const [name, setName] = useState("")
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState("")
  const submit = async () => {
    if (!name.trim() || saving) return
    setSaving(true)
    setError("")
    try {
      await action.onSubmit(name.trim())
      onDone()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not create this item")
      setSaving(false)
    }
  }
  return <div style={{ marginLeft: INDENT_BASE + depth * INDENT_STEP + 20 }}>
    <input autoFocus aria-label={action.placeholder} placeholder={action.placeholder} value={name} disabled={saving}
      onChange={(event) => setName(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault()
          void submit()
        }
        if (event.key === "Escape") {
          event.stopPropagation()
          onCancel()
        }
      }}
      className="h-6 w-full bg-transparent text-xs text-ink placeholder:text-ink-faint outline-none"
    />
    {error && <p role="alert" className="py-1 text-xs text-danger">{error}</p>}
  </div>
}
