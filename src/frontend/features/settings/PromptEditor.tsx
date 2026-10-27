"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Plus, Trash2, Save, X, RefreshCw } from "lucide-react"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { cn, isModKey } from "@/lib/utils"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { fetchPromptList, fetchPromptBlocks, fetchPromptRaw, savePrompt, deletePrompt, savePromptOrder, fetchPrompts } from "@/lib/backend/api"
import type { PromptMeta } from "@/lib/domain/types"

interface PromptEditorProps {
  open: boolean
  onClose: () => void
  onPromptsChanged: (prompts: Record<string, string>) => void
}

/** Lowercase kebab-case slug, capped at 64 chars. */
function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64) || "prompt"
}

interface PromptLists {
  list: PromptMeta[]
  blk: Record<string, string>
}

/** Fetch the prompt list together with the reusable `{{block}}` snippets. */
async function fetchPromptLists(): Promise<PromptLists> {
  const [list, blk] = await Promise.all([fetchPromptList(), fetchPromptBlocks()])
  return { list, blk }
}

/** Floating window to edit, reorder, create and delete prompt files, with `{{block}}` insertion. */
export function PromptEditor({ open, onClose, onPromptsChanged }: PromptEditorProps) {
  const { schedule, pending } = useUndoDelete()
  const [prompts, setPrompts] = useState<PromptMeta[]>([])
  const [blocks, setBlocks] = useState<Record<string, string>>({})
  const [selected, setSelected] = useState<string | null>(null)
  const [content, setContent] = useState("")
  const [dirty, setDirty] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState("")
  useEscapeLayer(() => setCreating(false), creating)
  const dragSlug = useRef<string | null>(null)
  // Mirror of `prompts` so a drop right after a reorder reads the newest order outside any state updater.
  const order = useRef<PromptMeta[]>([])

  const showPrompts = useCallback((next: PromptMeta[]) => {
    order.current = next
    setPrompts(next)
  }, [])

  // Live-reorder the list as the dragged item passes over others; persist once on drop.
  const reorder = useCallback((overSlug: string) => {
    const from = dragSlug.current
    if (!from || from === overSlug) return
    const fi = order.current.findIndex(p => p.slug === from)
    const ti = order.current.findIndex(p => p.slug === overSlug)
    if (fi < 0 || ti < 0) return
    const next = [...order.current]
    next.splice(ti, 0, next.splice(fi, 1)[0])
    showPrompts(next)
  }, [showPrompts])

  const persistOrder = useCallback(() => {
    dragSlug.current = null
    void savePromptOrder(order.current.map(p => p.slug))
  }, [])

  const applyLists = useCallback(({ list, blk }: PromptLists) => {
    showPrompts(list)
    setBlocks(blk)
  }, [showPrompts])

  /** Reload the lists and, with `notify`, tell the owner the resolved prompts changed. Both fetches run together, the lists apply as soon as they land. */
  const reload = useCallback(async (notify: boolean) => {
    const resolved = notify ? fetchPrompts() : null
    // Awaited below, the no-op handler only keeps a rejection from going unhandled when the lists fail first.
    resolved?.catch(() => {})
    applyLists(await fetchPromptLists())
    if (resolved) onPromptsChanged(await resolved)
  }, [applyLists, onPromptsChanged])

  useEffect(() => {
    if (open) fetchPromptLists().then(applyLists)
  }, [open, applyLists])

  const load = useCallback(async (slug: string) => {
    const { content: raw } = await fetchPromptRaw(slug)
    setSelected(slug)
    setContent(raw)
    setDirty(false)
    setCreating(false)
  }, [])

  const handleSave = useCallback(async () => {
    if (!selected) return
    await savePrompt(selected, content)
    setDirty(false)
    await reload(true)
  }, [selected, content, reload])

  const handleDelete = useCallback(() => {
    if (!selected) return
    const slug = selected
    schedule(`prompt:${slug}`, prompts.find((prompt) => prompt.slug === slug)?.name ?? slug, async () => {
      await deletePrompt(slug)
      await reload(true)
    })
    setSelected(null)
    setContent("")
    setDirty(false)
  }, [selected, prompts, schedule, reload])
  const handleCreate = useCallback(async () => {
    if (!newName.trim()) return
    const slug = slugify(newName)
    // Starting frontmatter for a new prompt
    const initial = `---\nname: ${newName.trim()}\nincludes:\n---\n\n`
    await savePrompt(slug, initial)
    setCreating(false)
    setNewName("")
    await reload(true)
    await load(slug)
  }, [newName, reload, load])

  const insertBlock = useCallback((blockName: string) => {
    setContent(prev => prev + `\n{{${blockName}}}`)
    setDirty(true)
  }, [])

  if (!open) return null

  return (
    <FloatingWindow onClose={onClose}>
      <div className="flex h-full">
        {/* Sidebar: prompt list */}
        <div className="w-56 shrink-0 border-r border-divider/50 flex flex-col">
          <div className="flex items-center justify-between p-3 border-b border-divider/50">
            <span className="text-xs font-medium text-ink-subtle">Prompts</span>
            <div className="flex gap-1">
              <button aria-label="Create prompt" onClick={() => { setCreating(true); setNewName("") }} className="p-1 rounded hover:bg-hover text-ink-muted hover:text-ink">
                <Plus className="h-3.5 w-3.5" />
              </button>
              <button aria-label="Refresh prompts" onClick={() => void reload(false)} className="p-1 rounded hover:bg-hover text-ink-muted hover:text-ink">
                <RefreshCw className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {creating && (
            <div className="p-2 border-b border-divider/50 flex gap-1">
              <input
                autoFocus
                value={newName}
                onChange={e => setNewName(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter") handleCreate() }}
                placeholder="Prompt name..."
                className="flex-1 rounded border border-divider bg-surface px-2 py-1 text-xs text-ink outline-none"
              />
              <button aria-label="Save new prompt" onClick={handleCreate} className="p-1 rounded hover:bg-hover text-ink-muted hover:text-ink">
                <Save className="h-3.5 w-3.5" />
              </button>
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {/* Hide prompts with a pending undoable delete */}
            {prompts.filter((prompt) => !pending(`prompt:${prompt.slug}`)).map(p => (
              <button
                key={p.slug}
                draggable
                onDragStart={() => { dragSlug.current = p.slug }}
                onDragEnter={() => reorder(p.slug)}
                onDragOver={e => e.preventDefault()}
                onDragEnd={persistOrder}
                onClick={() => load(p.slug)}
                className={cn(
                  "w-full text-left px-3 py-2 text-xs transition-colors truncate cursor-grab active:cursor-grabbing",
                  selected === p.slug ? "bg-surface text-ink" : "text-ink-muted hover:bg-hover hover:text-ink"
                )}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        {/* Main editor area */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex items-center justify-between p-3 border-b border-divider/50">
            <div className="flex items-center gap-2">
              {selected && (
                <>
                  <span className="text-sm font-medium text-ink">
                    {prompts.find(p => p.slug === selected)?.name ?? selected}
                  </span>
                  {dirty && <span className="text-[10px] text-ink-faint">(unsaved)</span>}
                </>
              )}
            </div>
            <div className="flex items-center gap-1">
              {selected && (
                <>
                  <button
                    onClick={handleSave}
                    disabled={!dirty}
                    className={cn("flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs transition-colors",
                      dirty ? "bg-surface-elevated text-ink hover:bg-hover-strong" : "text-ink-faint cursor-default"
                    )}
                  >
                    <Save className="h-3.5 w-3.5" /> Save
                  </button>
                  <button onClick={handleDelete} className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs text-danger hover:bg-danger-soft transition-colors">
                    <Trash2 className="h-3.5 w-3.5" /> Delete
                  </button>
                </>
              )}
              <button aria-label="Close prompt editor" onClick={onClose} className="p-1.5 rounded-lg hover:bg-hover text-ink-muted hover:text-ink ml-2">
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          {selected ? (
            <div className="flex-1 flex min-h-0">
              <textarea
                value={content}
                onChange={e => { setContent(e.target.value); setDirty(true) }}
                onKeyDown={e => { if (isModKey(e) && e.key === "s") { e.preventDefault(); handleSave() } }}
                spellCheck={false}
                className="flex-1 resize-none bg-transparent p-4 text-xs text-ink font-mono outline-none leading-relaxed"
                placeholder="Write your prompt..."
              />

              {/* Blocks sidebar */}
              <div className="w-44 shrink-0 border-l border-divider/50 flex flex-col">
                <div className="p-2 border-b border-divider/50">
                  <span className="text-[10px] font-medium text-ink-subtle uppercase tracking-wide">Blocks</span>
                </div>
                <div className="flex-1 overflow-y-auto p-1">
                  {Object.keys(blocks).map(name => (
                    <button
                      key={name}
                      onClick={() => insertBlock(name)}
                      className="w-full text-left px-2 py-1.5 text-xs text-ink-muted hover:text-ink hover:bg-hover rounded transition-colors truncate"
                    >
                      {`{{${name}}}`}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-ink-faint text-sm">
              Select a prompt to edit, or create a new one
            </div>
          )}
        </div>
      </div>
    </FloatingWindow>
  )
}
