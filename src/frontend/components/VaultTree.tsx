"use client"

import { createContext, useContext, useEffect, useState, type ElementType } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  ChevronRight,
  Folder,
  LayoutDashboard,
  Plus,
  Trash2,
  X,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Skeleton } from "./Skeleton"

export type VaultTreeItemType = "vault" | "folder" | "element"

export interface VaultTreeItem<T = unknown> {
  id: string
  label: string
  type: VaultTreeItemType
  icon?: ElementType
  data?: T
  children?: VaultTreeItem<T>[]
  isActive?: boolean
  isSystem?: boolean
  displayInVault?: boolean
  badge?: string | number
}

interface VaultTreeProps<T> {
  items?: VaultTreeItem<T>[]
  depth?: number
  loading?: boolean

  // Section header (root level only)
  sectionIcon?: ElementType
  sectionTitle?: string
  count?: number
  onPlusClick?: () => void
  plusTitle?: string
  collapsed?: boolean
  open?: boolean
  onOpenChange?: (open: boolean) => void
  onExpand?: () => void

  // Tree actions
  onVaultClick?: (id: string) => void
  onVaultDelete?: (id: string) => void
  onElementClick?: (item: VaultTreeItem<T>) => void
  onElementDelete?: (item: VaultTreeItem<T>) => void
  renderElement?: (item: VaultTreeItem<T>, depth: number) => React.ReactNode
  onAddVault?: (name: string) => Promise<void>
  onAddFolder?: (parentId: string | null, name: string) => Promise<void>
  onMoveElement?: (elementId: string, targetId: string | null) => Promise<void>
  onDashboardClick?: (vaultId: string) => void
  storageKey?: string
}

const INDENT_BASE = 8
const INDENT_STEP = 16

/* ─── context ─── */

interface TreeCtx<T> {
  expandedFolders: Set<string>
  toggleFolder: (id: string) => void
  dragOverId: string | null
  onDragOver: (e: React.DragEvent, targetId: string) => void
  onDragLeave: () => void
  onDrop: (e: React.DragEvent, targetId: string | null) => void
  onDragStart: (e: React.DragEvent, itemId: string) => void
  onVaultClick?: (id: string) => void
  onVaultDelete?: (id: string) => void
  onElementClick?: (item: VaultTreeItem<T>) => void
  onElementDelete?: (item: VaultTreeItem<T>) => void
  renderElement?: (item: VaultTreeItem<T>, depth: number) => React.ReactNode
  onAddFolder?: (parentId: string | null, name: string) => Promise<void>
  onDashboardClick?: (vaultId: string) => void
  createMode: "vault" | "folder" | null
  createParentId: string | null
  createName: string
  setCreateMode: (mode: "vault" | "folder" | null) => void
  setCreateParentId: (id: string | null) => void
  setCreateName: (name: string) => void
  handleCreateSubmit: () => Promise<void>
}

const TreeCtx = createContext<TreeCtx<unknown> | null>(null)

export function useTree<T = unknown>() {
  const ctx = useContext(TreeCtx)
  if (!ctx) throw new Error("useTree must be inside VaultTree")
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

/* ─── vault tree (public) ─── */

export function VaultTree<T>({
  items,
  depth = 0,
  loading,
  sectionIcon: SectionIcon,
  sectionTitle,
  count,
  onPlusClick,
  plusTitle,
  collapsed,
  open,
  onOpenChange,
  onExpand,
  onVaultClick,
  onVaultDelete,
  onElementClick,
  onElementDelete,
  renderElement,
  onAddVault,
  onAddFolder,
  onMoveElement,
  onDashboardClick,
  storageKey,
}: VaultTreeProps<T>) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
  const [createMode, setCreateMode] = useState<"vault" | "folder" | null>(null)
  const [createParentId, setCreateParentId] = useState<string | null>(null)
  const [createName, setCreateName] = useState("")
  const [dragOverId, setDragOverId] = useState<string | null>(null)

  const isRoot = depth === 0

  useEffect(() => {
    if (storageKey) {
      try {
        const saved = localStorage.getItem(storageKey)
        if (saved) {
          setExpandedFolders(new Set(JSON.parse(saved)))
        }
      } catch {}
    }
  }, [storageKey])

  const updateExpanded = (updater: (prev: Set<string>) => Set<string>) => {
    setExpandedFolders((prev) => {
      const next = updater(prev)
      if (storageKey) {
        try {
          localStorage.setItem(storageKey, JSON.stringify(Array.from(next)))
        } catch {}
      }
      return next
    })
  }

  const toggleFolder = (id: string) => {
    updateExpanded((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const handleCreateSubmit = async () => {
    if (!createName.trim()) return
    if (createMode === "vault") {
      await onAddVault?.(createName.trim())
    } else if (createMode === "folder") {
      await onAddFolder?.(createParentId, createName.trim())
    }
    setCreateName("")
    setCreateMode(null)
    setCreateParentId(null)
    if (createParentId) {
      updateExpanded((prev) => {
        const next = new Set(prev)
        next.add(createParentId!)
        return next
      })
    }
  }

  const handleDragStart = (e: React.DragEvent, itemId: string) => {
    e.dataTransfer.setData("text/plain", itemId)
    e.dataTransfer.effectAllowed = "move"
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
    const elementId = e.dataTransfer.getData("text/plain")
    if (elementId && onMoveElement) onMoveElement(elementId, targetId)
  }

  const handlePlus = () => {
    if (onPlusClick) {
      onPlusClick()
    } else if (onAddVault) {
      setCreateMode("vault")
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
    expandedFolders,
    toggleFolder,
    dragOverId,
    onDragOver: handleDragOver,
    onDragLeave: handleDragLeave,
    onDrop: handleDrop,
    onDragStart: handleDragStart,
    onVaultClick,
    onVaultDelete,
    onElementClick,
    onElementDelete,
    renderElement,
    onAddFolder,
    onDashboardClick,
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
        title={sectionTitle}
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
            <button
              onClick={() => onOpenChange?.(!open)}
              className="flex-1 flex items-center gap-3 p-1 rounded-md text-left transition-colors hover:text-ink"
            >
              <SectionIcon className="h-4 w-4 shrink-0 text-ink-muted" />
              <span className="text-sm font-medium">{sectionTitle}</span>
              {count != null && count > 0 && (
                <span className="rounded-full bg-surface-elevated px-1.5 py-0.5 text-[10px] text-ink-subtle font-semibold">
                  {count}
                </span>
              )}
              {open != null && (
                <div className="ml-auto">
                  <ChevronIcon open={open} />
                </div>
              )}
            </button>
            {(onPlusClick || onAddVault || onAddFolder) && (
              <button
                onClick={(e) => { e.stopPropagation(); handlePlus() }}
                className="p-1 rounded text-ink-subtle hover:text-ink hover:bg-surface transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
                title={plusTitle}
              >
                <Plus className="h-4 w-4" />
              </button>
            )}
          </div>

          {open && (
            <>
              {(createMode === "vault" || createMode === "folder") && (
                <InlineCreateForm
                  depth={0}
                  value={createName}
                  onChange={setCreateName}
                  onSubmit={handleCreateSubmit}
                  onCancel={() => { setCreateMode(null); setCreateName("") }}
                  placeholder={createMode === "vault" ? "Project name" : "Folder name"}
                />
              )}
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

function TreeContent<T>({ items, depth }: { items?: VaultTreeItem<T>[]; depth: number }) {
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
        item.type === "vault" || item.type === "folder" ? (
          <BranchNode key={item.id} item={item} depth={depth} />
        ) : (
          <ElementNode key={item.id} item={item} depth={depth} />
        )
      )}
    </div>
  )
}

/* ─── branch node (vault + folder merged) ─── */

function BranchNode<T>({ item, depth }: { item: VaultTreeItem<T>; depth: number }) {
  const {
    expandedFolders,
    toggleFolder,
    dragOverId,
    onDragOver,
    onDragLeave,
    onDrop,
    onVaultClick,
    onVaultDelete,
    onAddFolder,
    onDashboardClick,
    createMode,
    createParentId,
    createName,
    setCreateMode,
    setCreateParentId,
    setCreateName,
    handleCreateSubmit,
  } = useTree<T>()

  const isVault = item.type === "vault"
  const isExpanded = expandedFolders.has(item.id)
  const BranchIcon = item.icon || Folder
  const hasChildren = item.children && item.children.length > 0
  const paddingLeft = INDENT_BASE + depth * INDENT_STEP

  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-md transition-all duration-200 ease-out group",
          isVault && item.isActive && "bg-surface-elevated border border-divider-strong",
          isVault && !item.isActive && "border border-transparent",
          dragOverId === item.id && "ring-1 ring-ink-muted bg-surface-elevated",
        )}
        onDragOver={(e) => { e.stopPropagation(); onDragOver(e, item.id) }}
          onDragLeave={(e) => { e.stopPropagation(); onDragLeave() }}
          onDrop={(e) => { e.stopPropagation(); onDrop(e, item.id) }}
        style={{ paddingLeft, paddingRight: 8 }}
      >
        <button
          onClick={() => {
            if (isVault) {
              onVaultClick?.(item.id)
              if (item.isActive) {
                if (isExpanded) toggleFolder(item.id)
              } else {
                if (!isExpanded) toggleFolder(item.id)
              }
            } else {
              toggleFolder(item.id)
            }
          }}
          className={cn(
            "flex items-center gap-1.5 py-1 flex-1 text-left transition-colors",
            !isVault && "text-ink-subtle hover:text-ink",
          )}
        >
          <ChevronIcon open={isExpanded} />
          <BranchIcon className={cn(
            "h-3.5 w-3.5 shrink-0",
            isVault && item.isActive ? "text-ink" : "text-ink-muted",
          )} />
          <span className={cn(
            "text-[11px] font-medium truncate",
            isVault && item.isActive ? "text-ink font-semibold" : "text-ink-muted hover:text-ink",
          )}>
            {item.label}
          </span>
          {item.badge != null && (
            <span className="text-[10px] text-ink-faint shrink-0">{item.badge}</span>
          )}
        </button>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          {isVault && onDashboardClick && (
            <button
              onClick={(e) => { e.stopPropagation(); onDashboardClick(item.id) }}
              className="p-0.5 rounded text-ink-faint hover:text-ink transition-colors"
              title="Dashboard"
            >
              <LayoutDashboard className="h-3 w-3" />
            </button>
          )}
          {onAddFolder && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                setCreateMode("folder")
                setCreateParentId(item.id)
                setCreateName("")
              }}
              className="p-0.5 rounded text-ink-faint hover:text-ink transition-colors"
              title="New folder"
            >
              <Plus className="h-3 w-3" />
            </button>
          )}
          {onVaultDelete && (
            <button
              onClick={(e) => { e.stopPropagation(); onVaultDelete(item.id) }}
              className="p-0.5 rounded text-ink-faint hover:text-danger transition-colors"
              title={isVault ? "Delete project" : "Delete folder"}
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

      {createMode === "folder" && createParentId === item.id && (
        <InlineCreateForm
          depth={depth + 1}
          value={createName}
          onChange={setCreateName}
          onSubmit={handleCreateSubmit}
          onCancel={() => { setCreateMode(null); setCreateName("") }}
          placeholder="Folder name"
        />
      )}
    </div>
  )
}

/* ─── element node ─── */

function ElementNode<T>({ item, depth }: { item: VaultTreeItem<T>; depth: number }) {
  const { onDragStart, onElementClick, onElementDelete, onDashboardClick, renderElement } = useTree<T>()
  const paddingLeft = INDENT_BASE + depth * INDENT_STEP
  const ElementIcon = item.icon

  if (renderElement) {
    return <>{renderElement(item, depth)}</>
  }

  const handleClick = () => {
    if (item.isSystem && onDashboardClick && item.data) {
      onDashboardClick(item.data as string)
    } else {
      onElementClick?.(item)
    }
  }

  return (
    <div style={{ paddingLeft }}>
      <div
        className="flex items-center gap-1 py-0.5 group"
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
            <button
              onClick={() => onElementDelete(item)}
              className="p-0.5 rounded text-ink-faint hover:text-danger transition-colors shrink-0"
              title="Delete"
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
}: {
  depth: number
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  onCancel: () => void
  placeholder: string
}) {
  return (
    <div className="flex items-center gap-1 py-1" style={{ paddingLeft: INDENT_BASE + depth * INDENT_STEP }}>
      <input
        autoFocus
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onSubmit()
          if (e.key === "Escape") onCancel()
        }}
        className="flex-1 bg-surface/50 rounded px-1.5 py-0.5 text-[11px] text-ink outline-none focus:border-ink-muted transition-colors"
        placeholder={placeholder}
      />
      <button onClick={onSubmit} className="text-[10px] text-ink hover:text-ink">
        Create
      </button>
      <button onClick={onCancel} className="text-ink-faint hover:text-ink">
        <X className="h-3 w-3" />
      </button>
    </div>
  )
}
