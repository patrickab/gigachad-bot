"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { motion } from "framer-motion"
import { Check, ChevronDown, ChevronRight, FolderOpen, Globe, GripVertical, Loader2, Plus, X } from "lucide-react"
import { FloatingWindow } from "@/components/FloatingWindow"
import { LaTeXMarkdown } from "@/components/LaTeXMarkdown"
import { MemoryBoard } from "@/components/MemoryBoard"
import { AddMemoryCard, renderMemoriesMarkdown } from "@/components/MemoryPanel"
import { commitMemoryDoc, getCategories, getMemories, remapOrphanedCategory, saveCategories } from "@/lib/api"
import { sortMemoriesByCategoryOrder } from "@/lib/memoryUtils"
import { useMemoryViewer } from "@/contexts/MemoryViewerContext"
import type { CategoryDef, PreviewMemory } from "@/lib/types"

export function MemoryViewer() {
  const { target, closeMemoryViewer } = useMemoryViewer()
  if (!target) return null
  return <MemoryViewerInner key={`${target.scope}:${target.projectSlug ?? ""}`} onClose={closeMemoryViewer} target={target} />
}

// ---------------------------------------------------------------------------
// Confirmation popup for category deletion with memories
// ---------------------------------------------------------------------------

function DeleteCategoryPopup({
  category, memoryCount, remapping, onRemap, onDrop, onCancel,
}: {
  category: CategoryDef; memoryCount: number; remapping: boolean
  onRemap: () => void; onDrop: () => void; onCancel: () => void
}) {
  return (
    <div className="absolute inset-x-0 top-0 z-10 mx-3 mt-2 rounded-lg border border-divider bg-surface-elevated p-3 shadow-lg">
      <div className="text-xs font-medium text-ink">
        &ldquo;{category.name}&rdquo; has {memoryCount} {memoryCount === 1 ? "memory" : "memories"}
      </div>
      <div className="mt-0.5 text-[11px] text-ink-muted">Remap to remaining categories via LLM?</div>
      <div className="mt-2.5 flex items-center justify-end gap-2">
        <button onClick={onCancel} disabled={remapping}
          className="text-[11px] text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">Cancel</button>
        <button onClick={onDrop} disabled={remapping}
          className="rounded-md border border-divider px-2.5 py-1 text-[11px] text-ink-muted hover:bg-hover hover:text-ink disabled:opacity-40 transition-colors">
          Delete memories
        </button>
        <button onClick={onRemap} disabled={remapping}
          className="flex items-center gap-1 rounded-md border border-divider-strong bg-surface px-2.5 py-1 text-[11px] font-medium text-ink hover:bg-hover disabled:opacity-40 transition-colors">
          {remapping && <Loader2 className="h-3 w-3 animate-spin" />}
          Remap
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Add-category inline form
// ---------------------------------------------------------------------------

function AddCategoryForm({ existingNames, onAdd, onCancel }: {
  existingNames: Set<string>; onAdd: (cat: CategoryDef) => void; onCancel: () => void
}) {
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const nameRef = useRef<HTMLInputElement>(null)
  useEffect(() => { nameRef.current?.focus() }, [])

  const nameKey = name.trim().toLowerCase().replace(/\s+/g, "_")
  const isDuplicate = existingNames.has(nameKey)
  const canSave = name.trim().length > 0 && !isDuplicate

  function handleSave() {
    if (!canSave) return
    onAdd({ name: nameKey, description: description.trim() })
  }

  return (
    <div className="rounded-lg border border-divider bg-surface p-2.5 space-y-1.5">
      <input ref={nameRef} value={name} onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") handleSave(); if (e.key === "Escape") onCancel() }}
        placeholder="category_name"
        className="w-full rounded border border-divider bg-transparent px-2 py-1 text-xs text-ink placeholder:text-ink-faint outline-none focus:border-divider-strong" />
      {isDuplicate && <div className="text-[10px] text-red-400">Name already exists</div>}
      <input value={description} onChange={(e) => setDescription(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") handleSave(); if (e.key === "Escape") onCancel() }}
        placeholder="Short description (optional)"
        className="w-full rounded border border-divider bg-transparent px-2 py-1 text-xs text-ink placeholder:text-ink-faint outline-none focus:border-divider-strong" />
      <div className="flex items-center justify-end gap-2 pt-0.5">
        <button onClick={onCancel} className="text-[11px] text-ink-muted hover:text-ink transition-colors">Cancel</button>
        <button onClick={handleSave} disabled={!canSave}
          className="rounded-md border border-divider-strong bg-surface-elevated px-2.5 py-1 text-[11px] font-medium text-ink hover:bg-hover disabled:opacity-40 transition-colors">
          Add
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Categories panel — collapsible pill list with drag-to-reorder
// ---------------------------------------------------------------------------

function CategoriesPanel({
  categories, memories, disabled, onReorder, onDeleteCategory, onAddCategory,
}: {
  categories: CategoryDef[]; memories: PreviewMemory[]; disabled: boolean
  onReorder: (cats: CategoryDef[]) => void
  onDeleteCategory: (cat: CategoryDef) => void
  onAddCategory: (cat: CategoryDef) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [addingCat, setAddingCat] = useState(false)
  const [dragIdx, setDragIdx] = useState<number | null>(null)
  const [overIdx, setOverIdx] = useState<number | null>(null)
  const existingNames = new Set(categories.map((c) => c.name))

  function handleDrop(targetIdx: number) {
    if (dragIdx === null || dragIdx === targetIdx) return
    const next = [...categories]
    const [moved] = next.splice(dragIdx, 1)
    next.splice(targetIdx, 0, moved)
    onReorder(next)
    setDragIdx(null)
    setOverIdx(null)
  }

  return (
    <div className="border-b border-divider/30 shrink-0">
      <div className="flex items-center justify-between px-3 py-1.5">
        <button onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint hover:text-ink-subtle transition-colors">
          {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          Categories ({categories.length})
        </button>
        <button onClick={() => { setExpanded(true); setAddingCat(true) }} disabled={disabled}
          className="rounded-md p-1 text-ink-subtle hover:bg-hover hover:text-ink disabled:opacity-40 transition-colors" title="Add category">
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>
      {expanded && (
        <div className="px-3 pb-2 space-y-1.5">
          <div className="flex flex-wrap gap-1.5">
            {categories.map((cat, idx) => (
              <span
                key={cat.name}
                draggable={!disabled}
                onDragStart={() => { setDragIdx(idx) }}
                onDragOver={(e) => { e.preventDefault(); setOverIdx(idx) }}
                onDragLeave={() => setOverIdx(null)}
                onDrop={(e) => { e.preventDefault(); handleDrop(idx) }}
                onDragEnd={() => { setDragIdx(null); setOverIdx(null) }}
                className={[
                  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] text-ink-muted transition-all",
                  overIdx === idx && dragIdx !== idx
                    ? "border-divider-strong bg-hover scale-105"
                    : dragIdx === idx
                    ? "border-divider opacity-40"
                    : "border-divider bg-surface",
                ].join(" ")}
              >
                {!disabled && <GripVertical className="h-2.5 w-2.5 text-ink-faint cursor-grab" />}
                {cat.name.replace(/_/g, " ")}
                {!disabled && (
                  <button onClick={() => onDeleteCategory(cat)}
                    className="ml-0.5 rounded-full p-0.5 text-ink-faint hover:text-ink-muted transition-colors"
                    aria-label={`Remove category ${cat.name}`}>
                    <X className="h-2.5 w-2.5" />
                  </button>
                )}
              </span>
            ))}
            {categories.length === 0 && (
              <span className="text-[10px] italic text-ink-faint">No categories.</span>
            )}
          </div>
          {addingCat && (
            <AddCategoryForm existingNames={existingNames}
              onAdd={(cat) => { onAddCategory(cat); setAddingCat(false) }}
              onCancel={() => setAddingCat(false)} />
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main viewer
// ---------------------------------------------------------------------------

function MemoryViewerInner({
  target, onClose,
}: {
  target: { scope: "global" | "project"; projectSlug?: string | null }
  onClose: () => void
}) {
  const { scope, projectSlug } = target
  const title = scope === "global" ? "Global Profile" : "Project Memory"
  const ScopeIcon = scope === "global" ? Globe : FolderOpen

  const [memories, setMemories] = useState<PreviewMemory[]>([])
  const [categories, setCategories] = useState<CategoryDef[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const [pendingDeleteCat, setPendingDeleteCat] = useState<CategoryDef | null>(null)
  const [remapping, setRemapping] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true); setError(null)
    Promise.all([getMemories(scope, projectSlug ?? null), getCategories(scope, projectSlug ?? null)])
      .then(([memRes, catRes]) => {
        if (cancelled) return
        setMemories(memRes.memories)
        setCategories(catRes.categories)
      })
      .catch((e) => { if (!cancelled) setError((e as Error)?.message || "Failed to load memories") })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [scope, projectSlug])

  const handleEdit = useCallback((id: string, text: string) => {
    setMemories((prev) => prev.map((m) => m.id === id ? { ...m, text, updated_at: new Date().toISOString() } : m))
  }, [])

  const handleEditCategory = useCallback((id: string, category: string) => {
    setMemories((prev) => prev.map((m) => m.id === id ? { ...m, category, updated_at: new Date().toISOString() } : m))
  }, [])

  const handleRemove = useCallback((id: string) => {
    setMemories((prev) => prev.filter((m) => m.id !== id))
  }, [])

  const handleAdd = useCallback((s: "global" | "project", text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return
    const now = new Date().toISOString()
    setMemories((prev) => [...prev, {
      id: `manual-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      text: trimmed, category: categories[0]?.name ?? "note", scope: s,
      created_at: now, updated_at: now,
    }])
  }, [categories])

  const handleAddCategory = useCallback((cat: CategoryDef) => {
    setCategories((prev) => [...prev, cat])
  }, [])

  const handleReorderCategories = useCallback((next: CategoryDef[]) => {
    setCategories(next)
  }, [])

  const handleDeleteCategory = useCallback((cat: CategoryDef) => {
    if (memories.some((m) => m.category === cat.name)) {
      setPendingDeleteCat(cat)
    } else {
      setCategories((prev) => prev.filter((c) => c.name !== cat.name))
    }
  }, [memories])

  const handleConfirmRemap = useCallback(async () => {
    if (!pendingDeleteCat) return
    const orphaned = memories.filter((m) => m.category === pendingDeleteCat.name)
    const remaining = categories.filter((c) => c.name !== pendingDeleteCat.name)
    if (remaining.length === 0) {
      setMemories((prev) => prev.filter((m) => m.category !== pendingDeleteCat.name))
      setCategories(remaining); setPendingDeleteCat(null); return
    }
    setRemapping(true)
    try {
      const res = await remapOrphanedCategory(scope, orphaned, remaining, projectSlug ?? null)
      const remappedById = new Map(res.memories.map((m) => [m.id, m]))
      setMemories((prev) =>
        prev.map((m) => remappedById.get(m.id) ?? m).filter((m) => m.category !== pendingDeleteCat.name)
      )
      setCategories(remaining); setPendingDeleteCat(null)
    } catch (e) {
      setError((e as Error)?.message || "Remap failed")
    } finally { setRemapping(false) }
  }, [pendingDeleteCat, memories, categories, scope, projectSlug])

  const handleConfirmDrop = useCallback(() => {
    if (!pendingDeleteCat) return
    setMemories((prev) => prev.filter((m) => m.category !== pendingDeleteCat.name))
    setCategories((prev) => prev.filter((c) => c.name !== pendingDeleteCat.name))
    setPendingDeleteCat(null)
  }, [pendingDeleteCat])

  const handleSave = useCallback(async () => {
    if (loading || saving) return
    setSaving(true)
    try {
      // Reorder memories to match category order before committing so both the
      // stored JSON and the rendered markdown reflect the current category sequence.
      const ordered = sortMemoriesByCategoryOrder(memories, categories)
      await saveCategories(scope, categories, projectSlug ?? null)
      await commitMemoryDoc(scope, [], projectSlug ?? null, null, null, ordered)
      onClose()
    } catch (e) {
      setError((e as Error)?.message || "Failed to save memories")
    } finally { setSaving(false) }
  }, [loading, saving, scope, projectSlug, memories, categories, onClose])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tgt = e.target as HTMLElement | null
      const editable = !!tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA" || tgt.isContentEditable)
      if (e.key === "Escape") {
        if (pendingDeleteCat) { setPendingDeleteCat(null); return }
        e.preventDefault(); onClose(); return
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !editable) { e.preventDefault(); handleSave() }
    }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [onClose, handleSave, pendingDeleteCat])

  // Preview always reflects current category order.
  const previewMarkdown = renderMemoriesMarkdown(memories, title, categories).trimEnd() + "\n"
  const isLocked = loading || saving || remapping

  return (
    <FloatingWindow onClose={onClose} overlayClassName="z-[95]" panelClassName="h-[86vh] max-h-[840px]">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        className="h-full flex flex-col" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between border-b border-divider/50 px-4 py-2 shrink-0">
          <div className="flex items-center gap-2">
            {isLocked ? <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-muted" /> : <ScopeIcon className="h-3.5 w-3.5 text-ink-muted" />}
            <span className="text-xs font-medium text-ink-muted">{title}</span>
            <span className="text-[10px] text-ink-faint">{memories.length} entries</span>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={onClose} disabled={saving || remapping}
              className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
              <X className="h-3 w-3" />Close
            </button>
            <button onClick={handleSave} disabled={isLocked}
              className="flex items-center gap-1.5 rounded-md border border-divider-strong bg-surface-elevated px-3 py-1.5 text-xs font-medium text-ink hover:bg-hover disabled:opacity-40 transition-colors">
              {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
              Save
            </button>
          </div>
        </div>

        {loading ? (
          <div className="flex flex-1 items-center justify-center gap-2 min-h-0">
            <Loader2 className="h-4 w-4 animate-spin text-ink-muted" />
            <span className="text-xs text-ink-muted">Loading memories…</span>
          </div>
        ) : error ? (
          <div className="flex flex-1 items-center justify-center p-8 min-h-0">
            <div className="w-full max-w-lg rounded-xl border border-divider bg-surface p-5">
              <div className="text-sm font-medium text-ink">Memory error</div>
              <div className="mt-2 whitespace-pre-wrap text-xs leading-5 text-ink-muted">{error}</div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex min-h-0">
            {/* Left pane */}
            <div className="w-1/2 min-w-0 border-r border-divider/50 flex flex-col relative">
              {pendingDeleteCat && (
                <DeleteCategoryPopup
                  category={pendingDeleteCat}
                  memoryCount={memories.filter((m) => m.category === pendingDeleteCat.name).length}
                  remapping={remapping}
                  onRemap={handleConfirmRemap}
                  onDrop={handleConfirmDrop}
                  onCancel={() => setPendingDeleteCat(null)}
                />
              )}

              <CategoriesPanel
                categories={categories} memories={memories} disabled={isLocked}
                onReorder={handleReorderCategories}
                onDeleteCategory={handleDeleteCategory}
                onAddCategory={handleAddCategory}
              />

              <div className="flex items-center justify-between border-b border-divider/30 px-3 py-1.5 shrink-0">
                <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
                  <ScopeIcon className="h-3 w-3" />Memories
                </div>
                <button onClick={() => setAdding((v) => !v)} disabled={isLocked}
                  className="rounded-md p-1 text-ink-subtle hover:bg-hover hover:text-ink disabled:opacity-40 transition-colors" title="Add memory">
                  <Plus className="h-3.5 w-3.5" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4">
                {adding && (
                  <div className="mb-3">
                    <AddMemoryCard scope={scope} disabled={isLocked} onAddMemory={handleAdd} onCancel={() => setAdding(false)} />
                  </div>
                )}
                <MemoryBoard
                  memories={memories}
                  categoryOrder={categories}
                  scope={scope}
                  disabled={isLocked}
                  mode="display"
                  onChange={handleEdit}
                  onChangeCategory={handleEditCategory}
                  onRemove={handleRemove}
                />
              </div>
            </div>

            {/* Right pane — markdown preview */}
            <div className="w-1/2 min-w-0 flex flex-col">
              <div className="flex items-center gap-2 border-b border-divider/30 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
                Preview
              </div>
              <div className="flex-1 overflow-auto p-4">
                <LaTeXMarkdown content={previewMarkdown} compact />
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </FloatingWindow>
  )
}
