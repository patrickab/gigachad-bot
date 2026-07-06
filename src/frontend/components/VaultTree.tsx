"use client"

import { createContext, useContext, useState, type ElementType } from "react"
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
import type { VaultBranchController } from "@/hooks/useVaultTree"

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
  /** Vault accepts mountpoints — shows the mount (HardDrive) hover button. */
  mountable?: boolean
  /** This vault row is a mounted FileVault (not a native tree vault like a project). */
  mounted?: boolean
}

interface VaultTreeProps<T> {
  items?: VaultTreeItem<T>[]
  depth?: number
  loading?: boolean

  // Section header (root level only)
  sectionIcon?: ElementType
  sectionTitle?: string
  onPlusClick?: () => void
  plusTitle?: string
  collapsed?: boolean
  open?: boolean
  onOpenChange?: (open: boolean) => void
  onExpand?: () => void

  // Branch expansion + activation — owned by the shared primitive (see useVaultTree)
  controller: VaultBranchController

  // Tree actions
  onVaultDelete?: (id: string) => void
  onElementClick?: (item: VaultTreeItem<T>) => void
  onElementDelete?: (item: VaultTreeItem<T>) => void
  renderElement?: (item: VaultTreeItem<T>, depth: number) => React.ReactNode
  onAddVault?: (name: string) => Promise<void>
  vaultPlaceholder?: string
  onAddMountpoint?: (vaultId: string, path: string) => Promise<void>
  mountpointPlaceholder?: string
  onAddFolder?: (parentId: string | null, name: string) => Promise<void>
  folderPlaceholder?: string
  folderIcon?: ElementType
  onMoveElement?: (elementId: string, targetId: string | null) => Promise<void>
  onDashboardClick?: (vaultId: string) => void
  onMemoryClick?: (vaultId: string) => void
}

const INDENT_BASE = 8
const INDENT_STEP = 16

/* ─── context ─── */

interface TreeCtx<T> {
  controller: VaultBranchController
  dragOverId: string | null
  onDragOver: (e: React.DragEvent, targetId: string) => void
  onDragLeave: () => void
  onDrop: (e: React.DragEvent, targetId: string | null) => void
  onDragStart: (e: React.DragEvent, itemId: string) => void
  onVaultDelete?: (id: string) => void
  onElementClick?: (item: VaultTreeItem<T>) => void
  onElementDelete?: (item: VaultTreeItem<T>) => void
  renderElement?: (item: VaultTreeItem<T>, depth: number) => React.ReactNode
  onAddFolder?: (parentId: string | null, name: string) => Promise<void>
  folderPlaceholder: string
  folderIcon: ElementType
  onAddMountpoint?: (vaultId: string, path: string) => Promise<void>
  mountpointPlaceholder: string
  onDashboardClick?: (vaultId: string) => void
  onMemoryClick?: (vaultId: string) => void
  createMode: "vault" | "folder" | "mountpoint" | null
  createParentId: string | null
  createName: string
  setCreateMode: (mode: "vault" | "folder" | "mountpoint" | null) => void
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
  onPlusClick,
  plusTitle,
  collapsed,
  open,
  onOpenChange,
  onExpand,
  controller,
  onVaultDelete,
  onElementClick,
  onElementDelete,
  renderElement,
  onAddVault,
  vaultPlaceholder = "Project name",
  onAddMountpoint,
  mountpointPlaceholder = "Mountpoint path…",
  onAddFolder,
  folderPlaceholder = "Folder name",
  folderIcon = Folder,
  onMoveElement,
  onDashboardClick,
  onMemoryClick,
}: VaultTreeProps<T>) {
  const [createMode, setCreateMode] = useState<"vault" | "folder" | "mountpoint" | null>(null)
  const [createParentId, setCreateParentId] = useState<string | null>(null)
  const [createName, setCreateName] = useState("")
  const [dragOverId, setDragOverId] = useState<string | null>(null)

  const isRoot = depth === 0

  const handleCreateSubmit = async () => {
    if (!createName.trim()) return
    if (createMode === "vault") {
      await onAddVault?.(createName.trim())
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
    controller,
    dragOverId,
    onDragOver: handleDragOver,
    onDragLeave: handleDragLeave,
    onDrop: handleDrop,
    onDragStart: handleDragStart,
    onVaultDelete,
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
            <button
              onClick={() => onOpenChange?.(!open)}
              className="flex-1 flex items-center gap-3 p-1 rounded-md text-left transition-colors hover:text-ink"
            >
              <SectionIcon className="h-4 w-4 shrink-0 text-ink-muted" />
              <span className="text-sm font-medium">{sectionTitle}</span>
              {open != null && (
                <div className="ml-auto">
                  <ChevronIcon open={open} />
                </div>
              )}
            </button>
            {(onPlusClick || onAddVault || onAddFolder) && (
              <button
                onClick={(e) => { e.stopPropagation(); handlePlus() }}
                className="p-1 rounded text-ink-subtle hover:text-ink hover:bg-surface transition-colors"
              >
                <Plus className="h-4 w-4" />
              </button>
            )}
          </div>

          {open && (
            <>
              <AnimatePresence>
                {(createMode === "vault" || createMode === "folder") && (
                  <InlineCreateForm
                    key="root-create"
                    depth={0}
                    value={createName}
                    onChange={setCreateName}
                    onSubmit={handleCreateSubmit}
                    onCancel={() => { setCreateMode(null); setCreateName("") }}
                    placeholder={createMode === "vault" ? vaultPlaceholder : folderPlaceholder}
                    icon={createMode === "vault" ? undefined : folderIcon}
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
    controller,
    dragOverId,
    onDragOver,
    onDragLeave,
    onDrop,
    onVaultDelete,
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
  } = useTree<T>()

  const isVault = item.type === "vault"
  const isExpanded = controller.isExpanded(item.id)
  const BranchIcon = item.icon || Folder
  const hasChildren = item.children && item.children.length > 0
  const paddingLeft = INDENT_BASE + depth * INDENT_STEP

  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-md transition-colors duration-200 ease-out group",
          dragOverId === item.id && "ring-1 ring-ink-muted bg-surface-elevated",
        )}
        onDragOver={(e) => { e.stopPropagation(); onDragOver(e, item.id) }}
          onDragLeave={(e) => { e.stopPropagation(); onDragLeave() }}
          onDrop={(e) => { e.stopPropagation(); onDrop(e, item.id) }}
        style={{ paddingLeft, paddingRight: 8 }}
      >
        <button
          onClick={() => controller.toggleBranch(item.id)}
          className={cn(
            "flex items-center gap-1.5 py-1 flex-1 text-left transition-colors",
            isVault
              ? item.isActive ? "text-ink" : "text-ink-subtle hover:text-ink-muted"
              : "text-ink-subtle hover:text-ink",
          )}
        >
          <ChevronIcon open={isExpanded} />
          <BranchIcon className={cn(
            "h-3.5 w-3.5 shrink-0",
            isVault
              ? item.isActive ? "text-ink" : "text-ink-subtle"
              : "text-ink-muted",
          )} />
          <span className={cn(
            "text-[11px] font-medium truncate",
            isVault
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
          {isVault && !item.mounted && onMemoryClick && (
            <button
              onClick={(e) => { e.stopPropagation(); onMemoryClick(item.id) }}
              className="p-0.5 rounded text-ink-faint hover:text-ink transition-colors"
            >
              <Brain className="h-3 w-3" />
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
            >
              <Plus className="h-3 w-3" />
            </button>
          )}
          {isVault && item.mountable && onAddMountpoint && (
            <button
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
          {onVaultDelete && (
            <button
              onClick={(e) => { e.stopPropagation(); onVaultDelete(item.id) }}
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
