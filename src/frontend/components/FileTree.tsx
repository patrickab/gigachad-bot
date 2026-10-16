"use client"

import { createContext, useContext, useRef, useState, type ElementType } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Brain,
  Check,
  ChevronRight,
  Folder,
  HardDrive,
  Plus,
  Trash2,
  X,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Skeleton } from "./Skeleton"
import type { FileBranchController } from "@/hooks/useFileTree"

export type FileTreeItemType = "root" | "folder" | "element"

export interface FileTreeItem<T = unknown> {
  id: string
  label: string
  type: FileTreeItemType
  icon?: ElementType
  data?: T
  children?: FileTreeItem<T>[]
  isActive?: boolean
  isSystem?: boolean
  displayInRoot?: boolean
  badge?: string | number
  /** File root accepts mountpoints — shows the mount (HardDrive) hover button. */
  mountable?: boolean
  /** This file root row is a mounted FileRoots (not a native tree file root like a project). */
  mounted?: boolean
}

interface FileTreeProps<T> {
  items?: FileTreeItem<T>[]
  depth?: number
  loading?: boolean

  // Section header (root level only)
  sectionIcon?: ElementType
  sectionTitle?: string
  onPlusClick?: () => void
  plusTitle?: string
  collapsed?: boolean
  /** Section is collapsible only when `onOpenChange` is given; otherwise it is always open. */
  open?: boolean
  onOpenChange?: (open: boolean) => void
  onExpand?: () => void

  // Branch expansion + activation — owned by the shared primitive (see useFileTree)
  controller: FileBranchController

  // Tree actions
  onRootDelete?: (id: string) => void
  onElementClick?: (item: FileTreeItem<T>) => void
  onElementDelete?: (item: FileTreeItem<T>) => void
  renderElement?: (item: FileTreeItem<T>, depth: number) => React.ReactNode
  onAddRoot?: (name: string) => Promise<void>
  rootPlaceholder?: string
  onAddMountpoint?: (rootId: string, path: string) => Promise<void>
  mountpointPlaceholder?: string
  onAddFolder?: (parentId: string | null, name: string) => Promise<void>
  folderPlaceholder?: string
  folderIcon?: ElementType
  onMoveElement?: (elementId: string, targetId: string | null) => Promise<void>
  onDashboardClick?: (rootId: string) => void
  onMemoryClick?: (rootId: string) => void

  // Multi-selection: ctrl/cmd+click toggles, shift+click selects a range, selected rows drag together
  multiSelect?: boolean
  /** Selected ids (files and/or folders) dropped on a row; `null` = dropped on the tree background. */
  onDropFiles?: (ids: string[], targetId: string | null) => void
}

/** Drag payload type for multi-selected rows: a JSON array of item ids. */
export const FILE_DRAG_TYPE = "application/x-file-tree-ids"

const INDENT_BASE = 8
const INDENT_STEP = 16

/** Ids of the rows currently on screen, top to bottom — the order shift+click ranges over. */
function visibleItemIds<T>(items: FileTreeItem<T>[] | undefined, isExpanded: (id: string) => boolean): string[] {
  return (items ?? []).flatMap((item) => [
    item.id,
    ...(item.type !== "element" && isExpanded(item.id) ? visibleItemIds(item.children, isExpanded) : []),
  ])
}

/* ─── context ─── */

interface TreeCtx<T> {
  controller: FileBranchController
  dragOverId: string | null
  onDragOver: (e: React.DragEvent, targetId: string) => void
  onDragLeave: () => void
  onDrop: (e: React.DragEvent, targetId: string | null) => void
  onDragStart: (e: React.DragEvent, itemId: string) => void
  onRootDelete?: (id: string) => void
  onElementClick?: (item: FileTreeItem<T>) => void
  onElementDelete?: (item: FileTreeItem<T>) => void
  renderElement?: (item: FileTreeItem<T>, depth: number) => React.ReactNode
  onAddFolder?: (parentId: string | null, name: string) => Promise<void>
  folderPlaceholder: string
  folderIcon: ElementType
  onAddMountpoint?: (rootId: string, path: string) => Promise<void>
  mountpointPlaceholder: string
  onDashboardClick?: (rootId: string) => void
  onMemoryClick?: (rootId: string) => void
  multiSelect: boolean
  selectedIds: Set<string>
  /** Applies ctrl/shift selection; returns true when the click was a pure selection gesture. */
  onSelectClick: (item: FileTreeItem<T>, e: React.MouseEvent) => boolean
  createMode: "root" | "folder" | "mountpoint" | null
  createParentId: string | null
  createName: string
  setCreateMode: (mode: "root" | "folder" | "mountpoint" | null) => void
  setCreateParentId: (id: string | null) => void
  setCreateName: (name: string) => void
  handleCreateSubmit: () => Promise<void>
}

const TreeCtx = createContext<TreeCtx<unknown> | null>(null)

export function useTree<T = unknown>() {
  const ctx = useContext(TreeCtx)
  if (!ctx) throw new Error("useTree must be inside FileTree")
  return ctx as TreeCtx<T>
}

/* ─── chevron icon ─── */

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <motion.span
      animate={{ rotate: open ? 90 : 0 }}
      transition={{ duration: 0.15 }}
      className="shrink-0"
    >
      <ChevronRight className="h-3 w-3" />
    </motion.span>
  )
}

/* ─── file root tree (public) ─── */

export function FileTree<T>({
  items,
  depth = 0,
  loading,
  sectionIcon: SectionIcon,
  sectionTitle,
  onPlusClick,
  plusTitle,
  collapsed,
  open,
  onOpenChange,
  onExpand,
  controller,
  onRootDelete,
  onElementClick,
  onElementDelete,
  renderElement,
  onAddRoot,
  rootPlaceholder = "Project name",
  onAddMountpoint,
  mountpointPlaceholder = "Mountpoint path…",
  onAddFolder,
  folderPlaceholder = "Folder name",
  folderIcon = Folder,
  onMoveElement,
  onDashboardClick,
  onMemoryClick,
  multiSelect = false,
  onDropFiles,
}: FileTreeProps<T>) {
  const [createMode, setCreateMode] = useState<"root" | "folder" | "mountpoint" | null>(null)
  const [createParentId, setCreateParentId] = useState<string | null>(null)
  const [createName, setCreateName] = useState("")
  const [dragOverId, setDragOverId] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const selectionAnchorId = useRef<string | null>(null)

  const isRoot = depth === 0

  const handleCreateSubmit = async () => {
    if (!createName.trim()) return
    if (createMode === "root") {
      await onAddRoot?.(createName.trim())
    } else if (createMode === "folder") {
      await onAddFolder?.(createParentId, createName.trim())
    } else if (createMode === "mountpoint") {
      if (createParentId) await onAddMountpoint?.(createParentId, createName.trim())
    }
    setCreateName("")
    setCreateMode(null)
    setCreateParentId(null)
    if (createParentId) controller.expand(createParentId)
  }

  const handleDragStart = (e: React.DragEvent, itemId: string) => {
    if (multiSelect) {
      const draggedIds = selectedIds.has(itemId) ? [...selectedIds] : [itemId]
      if (!selectedIds.has(itemId)) setSelectedIds(new Set(draggedIds))
      e.dataTransfer.setData(FILE_DRAG_TYPE, JSON.stringify(draggedIds))
      e.dataTransfer.effectAllowed = "copyMove"
      return
    }
    e.dataTransfer.setData("text/plain", itemId)
    e.dataTransfer.effectAllowed = "move"
  }

  const handleSelectClick = (item: FileTreeItem<T>, e: React.MouseEvent): boolean => {
    if (!multiSelect) return false
    if (e.ctrlKey || e.metaKey) {
      const next = new Set(selectedIds)
      if (!next.delete(item.id)) next.add(item.id)
      setSelectedIds(next)
      selectionAnchorId.current = item.id
      return true
    }
    if (e.shiftKey) {
      const visibleIds = visibleItemIds(items, controller.isExpanded)
      const clickedIndex = visibleIds.indexOf(item.id)
      const anchorIndex = visibleIds.indexOf(selectionAnchorId.current ?? item.id)
      const [from, to] = anchorIndex < 0
        ? [clickedIndex, clickedIndex]
        : [Math.min(anchorIndex, clickedIndex), Math.max(anchorIndex, clickedIndex)]
      setSelectedIds(new Set(visibleIds.slice(from, to + 1)))
      return true
    }
    setSelectedIds(new Set([item.id]))
    selectionAnchorId.current = item.id
    return false
  }

  const handleDragOver = (e: React.DragEvent, targetId: string) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = "move"
    setDragOverId(targetId)
  }

  const handleDragLeave = () => { setDragOverId(null) }

  const handleDrop = (e: React.DragEvent, targetId: string | null) => {
    e.preventDefault()
    setDragOverId(null)
    const droppedIds = e.dataTransfer.getData(FILE_DRAG_TYPE)
    if (droppedIds) {
      onDropFiles?.(JSON.parse(droppedIds) as string[], targetId)
      return
    }
    const elementId = e.dataTransfer.getData("text/plain")
    if (elementId && onMoveElement) onMoveElement(elementId, targetId)
  }

  const handlePlus = () => {
    if (onPlusClick) {
      onPlusClick()
    } else if (onAddRoot) {
      setCreateMode("root")
      setCreateParentId(null)
      setCreateName("")
      onOpenChange?.(true)
    } else if (onAddFolder) {
      setCreateMode("folder")
      setCreateParentId(null)
      setCreateName("")
      onOpenChange?.(true)
    }
  }

  const ctx: TreeCtx<T> = {
    controller,
    dragOverId,
    onDragOver: handleDragOver,
    onDragLeave: handleDragLeave,
    onDrop: handleDrop,
    onDragStart: handleDragStart,
    onRootDelete,
    onElementClick,
    onElementDelete,
    renderElement,
    onAddFolder,
    folderPlaceholder,
    folderIcon,
    onAddMountpoint,
    mountpointPlaceholder,
    onDashboardClick,
    onMemoryClick,
    multiSelect,
    selectedIds,
    onSelectClick: handleSelectClick,
    createMode,
    createParentId,
    createName,
    setCreateMode,
    setCreateParentId,
    setCreateName,
    handleCreateSubmit,
  }

  // Collapsed (icon-only sidebar mode)
  if (collapsed && isRoot && SectionIcon) {
    return (
      <button
        onClick={() => { onExpand?.(); onOpenChange?.(true) }}
        className="w-full flex items-center justify-center p-2 rounded-md text-ink-muted hover:bg-surface-elevated/50 hover:text-ink transition-colors"
        aria-label={collapsed ? `Expand ${sectionTitle ?? "section"}` : undefined}
      >
        <SectionIcon className="h-4 w-4 shrink-0" />
      </button>
    )
  }

  // Loading state (root only)
  if (loading && isRoot) {
    return (
      <div className="space-y-1.5 px-2 py-1">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-3 w-32" />
        <Skeleton className="h-3 w-20" />
      </div>
    )
  }

  const content = (
    <TreeContent items={items} depth={depth} />
  )

  // Root header (only when section config is provided)
  if (isRoot && SectionIcon && sectionTitle != null) {
    return (
      <TreeCtx.Provider value={ctx as unknown as TreeCtx<unknown>}>
        <div className="space-y-1 w-full">
          <div className="flex w-full items-center justify-between p-1.5 rounded-md text-ink-muted hover:bg-surface-elevated/20 group">
            {onOpenChange ? (
              <button
                onClick={() => onOpenChange(!open)}
                className="flex-1 flex items-center gap-3 p-1 rounded-md text-left transition-colors hover:text-ink"
              >
                <SectionIcon className="h-4 w-4 shrink-0 text-ink-muted" />
                <span className="text-sm font-medium">{sectionTitle}</span>
                <div className="ml-auto">
                  <ChevronIcon open={!!open} />
                </div>
              </button>
            ) : (
              <div className="flex-1 flex items-center gap-3 p-1">
                <SectionIcon className="h-4 w-4 shrink-0 text-ink-muted" />
                <span className="text-sm font-medium">{sectionTitle}</span>
              </div>
            )}
            {(onPlusClick || onAddRoot || onAddFolder) && (
              <button aria-label={`${plusTitle ?? "Add item"}`}
                onClick={(e) => { e.stopPropagation(); handlePlus() }}
                className="p-1 rounded text-ink-subtle hover:text-ink hover:bg-surface transition-colors"
              >
                <Plus className="h-4 w-4" />
              </button>
            )}
          </div>

          {(open || !onOpenChange) && (
            <>
              <AnimatePresence>
                {(createMode === "root" || createMode === "folder") && (
                  <InlineCreateForm
                    key="root-create"
                    depth={0}
                    value={createName}
                    onChange={setCreateName}
                    onSubmit={handleCreateSubmit}
                    onCancel={() => { setCreateMode(null); setCreateName("") }}
                    placeholder={createMode === "root" ? rootPlaceholder : folderPlaceholder}
                    icon={createMode === "root" ? undefined : folderIcon}
                  />
                )}
              </AnimatePresence>
              {content}
            </>
          )}
        </div>
      </TreeCtx.Provider>
    )
  }

  // Non-root or no section config
  return (
    <TreeCtx.Provider value={ctx as unknown as TreeCtx<unknown>}>
      {content}
    </TreeCtx.Provider>
  )
}

/* ─── tree content (internal) ─── */

function TreeContent<T>({ items, depth }: { items?: FileTreeItem<T>[]; depth: number }) {
  const { onDragLeave, onDrop } = useTree<T>()
  const isRoot = depth === 0

  if (!items || items.length === 0) {
    return <p className="px-2 py-2 text-[11px] text-ink-faint">Empty</p>
  }

  return (
    <div
      className="space-y-0.5"
      onDragOver={isRoot ? (e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move" } : undefined}
      onDragLeave={isRoot ? () => onDragLeave() : undefined}
      onDrop={isRoot ? (e) => onDrop(e, null) : undefined}
    >
      {items.map((item) =>
        item.type === "root" || item.type === "folder" ? (
          <BranchNode key={item.id} item={item} depth={depth} />
        ) : (
          <ElementNode key={item.id} item={item} depth={depth} />
        )
      )}
    </div>
  )
}

/* ─── branch node (file root + folder merged) ─── */

function BranchNode<T>({ item, depth }: { item: FileTreeItem<T>; depth: number }) {
  const {
    controller,
    dragOverId,
    onDragOver,
    onDragLeave,
    onDrop,
    onRootDelete,
    onAddFolder,
    folderPlaceholder,
    folderIcon,
    onAddMountpoint,
    mountpointPlaceholder,
    onMemoryClick,
    createMode,
    createParentId,
    createName,
    setCreateMode,
    setCreateParentId,
    setCreateName,
    handleCreateSubmit,
    onDragStart,
    multiSelect,
    selectedIds,
    onSelectClick,
  } = useTree<T>()

  const isRootNode = item.type === "root"
  const isExpanded = controller.isExpanded(item.id)
  const BranchIcon = item.icon || Folder
  const hasChildren = item.children && item.children.length > 0
  const paddingLeft = INDENT_BASE + depth * INDENT_STEP

  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-md transition-colors duration-200 ease-out group",
          multiSelect && "select-none",
          selectedIds.has(item.id) && "bg-surface-elevated",
          dragOverId === item.id && "ring-1 ring-ink-muted bg-surface-elevated",
        )}
        onDragOver={(e) => { e.stopPropagation(); onDragOver(e, item.id) }}
          onDragLeave={(e) => { e.stopPropagation(); onDragLeave() }}
          onDrop={(e) => { e.stopPropagation(); onDrop(e, item.id) }}
        style={{ paddingLeft, paddingRight: 8 }}
      >
        <button
          draggable={multiSelect}
          onDragStart={multiSelect ? (e) => onDragStart(e, item.id) : undefined}
          onClick={(e) => { if (!onSelectClick(item, e)) controller.toggleBranch(item.id) }}
          className={cn(
            "flex items-center gap-1.5 py-1 flex-1 text-left transition-colors",
            isRootNode
              ? item.isActive ? "text-ink" : "text-ink-subtle hover:text-ink-muted"
              : "text-ink-subtle hover:text-ink",
          )}
        >
          <ChevronIcon open={isExpanded} />
          <BranchIcon className={cn(
            "h-3.5 w-3.5 shrink-0",
            isRootNode
              ? item.isActive ? "text-ink" : "text-ink-subtle"
              : "text-ink-muted",
          )} />
          <span className={cn(
            "text-[11px] font-medium truncate",
            isRootNode
              ? item.isActive ? "text-ink" : "text-ink-subtle"
              : undefined,
          )}>
            {item.label}
          </span>
          {item.badge != null && (
            <span className="text-[10px] text-ink-faint shrink-0">{item.badge}</span>
          )}
        </button>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          {isRootNode && !item.mounted && onMemoryClick && (
            <button aria-label={`Open memory for ${item.label}`}
              onClick={(e) => { e.stopPropagation(); onMemoryClick(item.id) }}
              className="p-0.5 rounded text-ink-faint hover:text-ink transition-colors"
            >
              <Brain className="h-3 w-3" />
            </button>
          )}
          {onAddFolder && (
            <button aria-label={`Add folder to ${item.label}`}
              onClick={(e) => {
                e.stopPropagation()
                setCreateMode("folder")
                setCreateParentId(item.id)
                setCreateName("")
              }}
              className="p-0.5 rounded text-ink-faint hover:text-ink transition-colors"
            >
              <Plus className="h-3 w-3" />
            </button>
          )}
          {isRootNode && item.mountable && onAddMountpoint && (
            <button aria-label={`Add mountpoint to ${item.label}`}
              onClick={(e) => {
                e.stopPropagation()
                setCreateMode("mountpoint")
                setCreateParentId(item.id)
                setCreateName("")
                controller.expand(item.id)
              }}
              className="p-0.5 rounded text-ink-faint hover:text-ink transition-colors"
            >
              <HardDrive className="h-3 w-3" />
            </button>
          )}
          {onRootDelete && (
            <button aria-label={`${item.mounted ? "Unmount" : "Delete"} ${item.label}`}
              onClick={(e) => { e.stopPropagation(); onRootDelete(item.id) }}
              className="p-0.5 rounded text-ink-faint hover:text-danger transition-colors"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && hasChildren && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden"
          >
            <TreeContent items={item.children} depth={depth + 1} />
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {(createMode === "folder" || createMode === "mountpoint") && createParentId === item.id && (
          <InlineCreateForm
            key={`${createMode}-create`}
            depth={depth + 1}
            value={createName}
            onChange={setCreateName}
            onSubmit={handleCreateSubmit}
            onCancel={() => { setCreateMode(null); setCreateName("") }}
            placeholder={createMode === "mountpoint" ? mountpointPlaceholder : folderPlaceholder}
            icon={createMode === "mountpoint" ? HardDrive : folderIcon}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

/* ─── element node ─── */

function ElementNode<T>({ item, depth }: { item: FileTreeItem<T>; depth: number }) {
  const { onDragStart, onElementClick, onElementDelete, onDashboardClick, renderElement, multiSelect, selectedIds, onSelectClick } = useTree<T>()
  const paddingLeft = INDENT_BASE + depth * INDENT_STEP
  const ElementIcon = item.icon

  if (renderElement) {
    return <>{renderElement(item, depth)}</>
  }

  const handleClick = (e: React.MouseEvent) => {
    if (onSelectClick(item, e)) return
    if (item.isSystem && onDashboardClick && item.data) {
      onDashboardClick(item.data as string)
    } else {
      onElementClick?.(item)
    }
  }

  return (
    <div style={{ paddingLeft }}>
      <div
        className={cn(
          "flex items-center gap-1 py-0.5 group rounded-md",
          multiSelect && "select-none",
          selectedIds.has(item.id) && "bg-surface-elevated",
        )}
        style={{ paddingRight: 8 }}
      >
        <button
          draggable
          onDragStart={(e) => onDragStart(e, item.id)}
          onClick={handleClick}
          className={cn(
            "flex items-center gap-1.5 flex-1 truncate text-left text-[11px] transition-colors",
            item.isActive ? "text-ink font-medium" : "text-ink-muted hover:text-ink",
          )}
        >
          {ElementIcon && <ElementIcon className="h-3.5 w-3.5 shrink-0 text-ink-muted" />}
          {item.label}
        </button>
          {onElementDelete && !item.isSystem && (
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <button aria-label={`Delete ${item.label}`}
              onClick={() => onElementDelete(item)}
              className="p-0.5 rounded text-ink-faint hover:text-danger transition-colors shrink-0"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

/* ─── inline create form ─── */

function InlineCreateForm({
  depth,
  value,
  onChange,
  onSubmit,
  onCancel,
  placeholder,
  icon: Icon = Folder,
}: {
  depth: number
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  onCancel: () => void
  placeholder: string
  icon?: ElementType
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.15, ease: "easeOut" }}
      className="flex items-center gap-1.5 py-1"
      style={{ paddingLeft: INDENT_BASE + depth * INDENT_STEP }}
    >
      <ChevronRight className="h-3 w-3 shrink-0 text-ink-muted" />
      <Icon className="h-3.5 w-3.5 shrink-0 text-ink-muted" />
      <input
        autoFocus
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onSubmit()
          if (e.key === "Escape") onCancel()
        }}
        className="flex-1 bg-transparent text-[11px] font-medium text-ink placeholder:text-ink-faint/50 outline-none"
        placeholder={placeholder}
      />
    </motion.div>
  )
}
