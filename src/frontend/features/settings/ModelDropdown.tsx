"use client"

import { ArrowLeft, Check, ChevronDown, ChevronRight, Cpu, Plus, Settings2, Trash2 } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { fetchOmpCatalog } from "@/lib/backend/api"
import { displayName, qualifiedModelIds } from "@/lib/models"
import type { ModelDefaults, ModelProvider, ModelsResponse, OmpCatalog } from "@/lib/domain/types"
import { cn } from "@/lib/utils"
import { PopoverSelector, PopoverSelectorAction } from "@/components/ui/PopoverSelector"
import { Skeleton } from "@/components/ui/Skeleton"
import { StyledSelect } from "@/components/ui/StyledSelect"

interface ProviderTab { label: string; models: string[] }

interface ModelDropdownProps {
  models: ModelsResponse | null
  loadError?: string | null
  selectedModel: string
  onSelect: (model: string) => void
  onProvidersChange: (providers: ModelProvider[]) => Promise<void>
  onTabOrderChange?: (order: string[]) => Promise<void>
  onDefaultsChange: (defaults: ModelDefaults) => Promise<void>
  /** Side of the trigger the panel opens on. */
  placement?: "above" | "below"
  /** Collapse the name when the expanded pill would cover this column. */
  chatColumn?: HTMLElement | null
}

function AddProvider({ onAdd }: { onAdd: (label: string, litellmId: string) => void }) {
  const [label, setLabel] = useState("")
  const [litellmId, setLitellmId] = useState("")
  const submit = () => onAdd(label, litellmId)

  return <div className="grid grid-cols-[1fr_1fr_auto] gap-1 border-t border-divider pt-2">
    <input autoFocus value={label} onChange={(event) => setLabel(event.target.value)} placeholder="Label" className="min-w-0 rounded border border-divider bg-paper px-2 py-1.5 text-xs text-ink outline-none" />
    <input value={litellmId} onChange={(event) => setLitellmId(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") submit() }} placeholder="LiteLLM ID" className="min-w-0 rounded border border-divider bg-paper px-2 py-1.5 text-xs text-ink outline-none" />
    <button aria-label="Add provider" onClick={submit} className="rounded p-1.5 text-ink-muted hover:bg-surface-elevated hover:text-ink"><Plus className="h-4 w-4" /></button>
  </div>
}

function ProviderRow({ provider, expanded, onToggle, onAddModel, onDeleteModel, onDeleteProvider }: {
  provider: ModelProvider
  expanded: boolean
  onToggle: () => void
  onAddModel: (model: string) => void
  onDeleteModel: (model: string) => void
  onDeleteProvider: () => void
}) {
  const [model, setModel] = useState("")
  const submit = () => { onAddModel(model); setModel("") }

  return <div className="group rounded-md hover:bg-surface-elevated">
    <div className="flex items-center">
      <button aria-label={`${expanded ? "Collapse" : "Expand"} ${provider.label}`} onClick={onToggle} className="p-2 text-ink-subtle hover:text-ink">{expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}</button>
      <button onClick={onToggle} className="flex min-w-0 flex-1 items-baseline gap-2 py-2 text-left"><span className="truncate text-xs text-ink">{provider.label}</span><span className="shrink-0 text-[10px] text-ink-subtle">{provider.models.length} {provider.models.length === 1 ? "model" : "models"}</span></button>
      <button aria-label={`Delete ${provider.label}`} onClick={onDeleteProvider} className="mr-1 rounded p-1.5 text-ink-subtle opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100 hover:text-danger"><Trash2 className="h-3.5 w-3.5" /></button>
    </div>
    {expanded && <div className="border-t border-divider px-2 pb-2 pt-1">
      {provider.models.map((item) => <div key={item} className="group flex items-center rounded-md px-2 py-1.5 text-xs text-ink hover:bg-surface-elevated"><span className="min-w-0 flex-1 truncate">{item}</span><button aria-label={`Delete ${item}`} onClick={() => onDeleteModel(item)} className="opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100 text-ink-subtle hover:text-danger"><Trash2 className="h-3.5 w-3.5" /></button></div>)}
      <div className="mt-1 flex gap-1 border-t border-divider pt-2"><input value={model} onChange={(event) => setModel(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") submit() }} aria-label="Model name" className="min-w-0 flex-1 rounded border border-divider bg-paper px-2 py-1.5 text-xs text-ink outline-none" /><button aria-label="Add model" onClick={submit} className="rounded px-2 text-ink-muted hover:bg-surface-elevated hover:text-ink"><Plus className="h-4 w-4" /></button></div>
    </div>}
  </div>
}

const OMP_SOURCE = "omp"
/** One catalog row holds every OMP pick, so the selector shows a single OMP
 *  tab listing `<omp-provider>/<model>` instead of a tab per login. */
const OMP_LABEL = "OMP"

/** Picks models out of OMP's own logins, including OAuth subscription plans.
 *
 * Every pick lands in the same provider catalog as a hand-entered provider:
 * one `OMP` row prefixed `litellm_proxy` so LiteLLM routes it at the OMP
 * gateway. Nothing downstream of the catalog needs to know OMP exists.
 * The picker still browses by login, because that is how OMP holds
 * credentials; only the destination row is shared.
 */
function OmpSection({ catalog, providers, onChange }: {
  catalog: OmpCatalog | null
  providers: ModelProvider[]
  onChange: (next: ModelProvider[]) => Promise<boolean>
}) {
  const [expanded, setExpanded] = useState<string | null>(null)
  if (!catalog?.installed) return null
  if (!catalog.online) return <p className="border-t border-divider px-3 py-2 text-[10px] text-ink-subtle">OMP is installed but its gateway is not answering on {catalog.gateway}. Start it with run-omp.sh to add subscription models.</p>

  const ompModels = providers.find((item) => item.label === OMP_LABEL)?.models ?? []
  const toggle = (modelId: string) => {
    const existing = providers.find((item) => item.label === OMP_LABEL)
    if (!existing) return onChange([...providers, { label: OMP_LABEL, litellm_id: catalog.litellm_id, source: OMP_SOURCE, models: [modelId] }])
    const models = existing.models.includes(modelId) ? existing.models.filter((value) => value !== modelId) : [...existing.models, modelId]
    return onChange(providers.map((item) => item.label === OMP_LABEL ? { ...item, models } : item))
  }

  return <div className="border-t border-divider">
    <p className="px-3 pb-1 pt-2 text-[10px] uppercase tracking-wide text-ink-subtle">From OMP</p>
    <div className="max-h-64 space-y-0.5 overflow-y-auto p-1.5 pt-0">
      {catalog.providers.map((provider) => {
        const picked = provider.models.filter((model) => ompModels.includes(model.id))
        const open = expanded === provider.id
        return <div key={provider.id} className="rounded-md hover:bg-surface-elevated">
          <button onClick={() => setExpanded(open ? null : provider.id)} className="flex w-full items-center">
            <span className="p-2 text-ink-subtle">{open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}</span>
            <span className="flex min-w-0 flex-1 items-baseline gap-2 py-2 text-left"><span className="truncate text-xs text-ink">{provider.label}</span><span className="shrink-0 text-[10px] text-ink-subtle">{picked.length} of {provider.models.length} added</span></span>
          </button>
          {open && <div className="border-t border-divider px-2 pb-2 pt-1">
            {provider.models.map((model) => {
              const added = ompModels.includes(model.id)
              return <button key={model.id} onClick={() => void toggle(model.id)} className={cn("flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-xs hover:bg-surface-elevated", added ? "text-ink" : "text-ink-muted")}>
                <span className="min-w-0 truncate">{displayName(model.id)}</span>
                {added ? <Check className="h-3.5 w-3.5 shrink-0" /> : <Plus className="h-3.5 w-3.5 shrink-0 text-ink-subtle" />}
              </button>
            })}
          </div>}
        </div>
      })}
    </div>
  </div>
}

function ProviderSettings({ providers, ompCatalog, onChange }: { providers: ModelProvider[]; ompCatalog: OmpCatalog | null; onChange: (next: ModelProvider[]) => Promise<boolean> }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const addProvider = async (labelInput: string, litellmIdInput: string) => {
    const label = labelInput.trim(), litellm_id = litellmIdInput.trim()
    if (!label || !litellm_id || providers.some((provider) => provider.label.toLowerCase() === label.toLowerCase())) return
    if (await onChange([...providers, { label, litellm_id, models: [] }])) setAdding(false)
  }
  return <>
    <div className="space-y-0.5 p-1.5">
      {providers.map((provider) => <ProviderRow key={provider.label} provider={provider} expanded={expanded === provider.label} onToggle={() => setExpanded(expanded === provider.label ? null : provider.label)} onAddModel={(model) => { const value = model.trim(); if (value && !provider.models.includes(value)) void onChange(providers.map((item) => item.label === provider.label ? { ...item, models: [...item.models, value] } : item)) }} onDeleteModel={(model) => void onChange(providers.map((item) => item.label === provider.label ? { ...item, models: item.models.filter((value) => value !== model) } : item))} onDeleteProvider={() => void onChange(providers.filter((item) => item.label !== provider.label))} />)}
      {adding ? <AddProvider onAdd={(label, id) => void addProvider(label, id)} /> : <button onClick={() => setAdding(true)} className="flex w-full items-center gap-1 rounded-md px-2 py-2 text-xs text-ink-muted hover:bg-surface-elevated hover:text-ink"><Plus className="h-3.5 w-3.5" /> Add provider</button>}
    </div>
    <OmpSection catalog={ompCatalog} providers={providers} onChange={onChange} />
  </>
}

function DefaultModels({ defaults, choices, onChange }: { defaults: ModelDefaults; choices: string[]; onChange: (next: ModelDefaults) => void }) {
  const labels: Record<keyof ModelDefaults, string> = { default_model: "Chat", small_model: "Small / fast", vision_model: "Vision / OCR", memory_model: "Memory", omp_model: "OMP invocation" }
  const options = choices.map((model) => ({ value: model, label: model }))
  return <div className="space-y-2 p-2">{(Object.keys(labels) as (keyof ModelDefaults)[]).map((key) => <label key={key} className="block"><span className="mb-1 block px-1 text-[10px] text-ink-subtle">{labels[key]}</span><StyledSelect ariaLabel={`${labels[key]} model`} value={defaults[key]} options={options} onChange={(model) => void onChange({ ...defaults, [key]: model })} /></label>)}</div>
}

/** Bare model name for the compact trigger: `litellm_proxy/anthropic/claude-sonnet-5` -> `claude-sonnet-5`. */
function shortName(model: string): string {
  return model.slice(model.lastIndexOf("/") + 1)
}

export function ModelDropdown({ models, loadError, selectedModel, onSelect, onProvidersChange, onTabOrderChange, onDefaultsChange, placement = "above", chatColumn }: ModelDropdownProps) {
  const [configuring, setConfiguring] = useState(false)
  const [settingsTab, setSettingsTab] = useState<"providers" | "defaults">("providers")
  const [error, setError] = useState<string | null>(null)
  const [draftDefaults, setDraftDefaults] = useState<ModelDefaults | null>(null)
  const [ompCatalog, setOmpCatalog] = useState<OmpCatalog | null>(null)
  const [compact, setCompact] = useState(false)
  const triggerRootRef = useRef<HTMLDivElement>(null)
  const labelRef = useRef<HTMLSpanElement>(null)
  const labelContentRef = useRef<HTMLSpanElement>(null)
  const loading = models === null

  useEffect(() => {
    const root = triggerRootRef.current
    const label = labelRef.current
    const content = labelContentRef.current
    const main = root?.closest("main")
    if (!chatColumn || !root || !label || !content || !main) return
    const measure = () => {
      // Keep overlap detection stable throughout the label's width animation.
      const expandedRight = root.getBoundingClientRect().right + content.getBoundingClientRect().width - label.getBoundingClientRect().width
      setCompact(expandedRight > chatColumn.getBoundingClientRect().left)
    }
    const observer = new ResizeObserver(measure)
    for (const element of [root, content, chatColumn, main]) observer.observe(element)
    measure()
    return () => observer.disconnect()
  }, [chatColumn, loading])
  // Default-model edits are buffered and flushed once on close, not saved per keystroke.
  const pendingDefaults = useRef<ModelDefaults | null>(null)
  const onClose = useCallback(() => {
    setConfiguring(false)
    const defaults = pendingDefaults.current
    pendingDefaults.current = null
    if (defaults) void onDefaultsChange(defaults).catch((reason) => setError((reason as Error).message || "Could not save default models"))
  }, [onDefaultsChange])
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape" && configuring) setConfiguring(false) }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [configuring])
  // Loaded when the settings panel opens, not on mount: OMP is optional, so
  // probing its gateway for every chat view would cost a request nobody asked
  // for. A failed probe stays null and simply hides the section.
  useEffect(() => {
    if (!configuring || ompCatalog) return
    fetchOmpCatalog().then(setOmpCatalog).catch(() => undefined)
  }, [configuring, ompCatalog])

  if (models === null) return loadError
    ? <span className="mx-2 text-xs text-ink-muted">Models unavailable</span>
    : <Skeleton className="mx-2 h-4 w-28" />

  const byDisplayName = (a: string, b: string) => displayName(a).localeCompare(displayName(b))
  const available: ProviderTab[] = [
    { label: "Ollama", models: [...models.ollama].sort(byDisplayName) },
    ...models.providers.map((provider) => ({ label: provider.label, models: qualifiedModelIds(provider).sort(byDisplayName) })),
  ]
  const selectableProviders = available.filter((provider) => provider.models.length > 0)
  const tabOrder = models.tab_order
  // Explicit position from the saved provider order. Unlisted providers (never
  // dragged, or new since the order was saved) fall to the end, alphabetical,
  // same rank rule PromptStore uses for prompt display order.
  const orderedProviders = [...selectableProviders].sort((a, b) => {
    const ai = tabOrder.indexOf(a.label)
    const bi = tabOrder.indexOf(b.label)
    if (ai !== -1 && bi !== -1) return ai - bi
    if (ai !== -1) return -1
    if (bi !== -1) return 1
    return a.label.localeCompare(b.label)
  })
  const choices = available.flatMap((provider) => provider.models)
  const persist = async (next: ModelProvider[]) => {
    setError(null)
    try { await onProvidersChange(next); return true } catch (reason) { setError((reason as Error).message || "Could not save providers"); return false }
  }
  const updateDefaults = (defaults: ModelDefaults) => { setDraftDefaults(defaults); pendingDefaults.current = defaults }
  const reorderProviders = (labels: string[]) => void onTabOrderChange?.(labels).catch((reason) => setError((reason as Error).message || "Could not save provider order"))
  const fullLabel = selectedModel ? displayName(selectedModel) : "Select model"
  const errorLine = error && <p className="px-2 pb-1.5 text-xs text-danger">{error}</p>

  const segmentClass = (active: boolean) => cn("flex-1 rounded-md px-2 py-1.5 text-xs transition-colors", active ? "bg-surface-elevated text-ink shadow-[var(--shadow-sm)]" : "text-ink-subtle hover:text-ink")
  const configurePanel = <>
    <div className="border-b border-divider p-1.5">
      <div className="flex rounded-lg bg-paper p-0.5">
        <button aria-label="Back to model selector" onClick={() => setConfiguring(false)} className="rounded-md px-2 py-1.5 text-ink-subtle hover:bg-surface-elevated hover:text-ink"><ArrowLeft className="h-3.5 w-3.5" /></button>
        <button onClick={() => setSettingsTab("providers")} className={segmentClass(settingsTab === "providers")}>Providers / Models</button>
        <button onClick={() => setSettingsTab("defaults")} className={segmentClass(settingsTab === "defaults")}>Default Models</button>
      </div>
    </div>
    {settingsTab === "providers"
      ? <ProviderSettings providers={models.providers} ompCatalog={ompCatalog} onChange={persist} />
      : <DefaultModels defaults={draftDefaults ?? models.defaults} choices={choices} onChange={updateDefaults} />}
    {errorLine}
  </>

  return <div ref={triggerRootRef} data-compact={Boolean(chatColumn && compact)} className="group/model min-w-0">
    <PopoverSelector
    triggerLabel={`Model: ${fullLabel}`}
    triggerClassName="h-8.5 gap-0 bg-paper px-[11px] hover:bg-surface-elevated"
    triggerContent={<>
      <Cpu className="h-3 w-3 shrink-0" />
      <span ref={labelRef} className="block max-w-[146px] overflow-hidden opacity-100 transition-[max-width,opacity] duration-250 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none group-[[data-compact=true]:not(:hover):not(:focus-within)]/model:max-w-0 group-[[data-compact=true]:not(:hover):not(:focus-within)]/model:opacity-0">
        <span ref={labelContentRef} className="block w-max pl-1.5">
          <span className="block max-w-[140px] truncate">{selectedModel ? shortName(selectedModel) : fullLabel}</span>
        </span>
      </span>
    </>}
    label="Model"
    categories={orderedProviders.map((provider) => ({ key: provider.label, label: provider.label, items: provider.models }))}
    itemKey={(model) => model}
    isSelected={(model) => model === selectedModel}
    renderItem={(model) => <span className="flex-1 truncate">{displayName(model)}</span>}
    onSelect={onSelect}
    onReorderCategories={reorderProviders}
    footer={<>{errorLine}<PopoverSelectorAction icon={Settings2} label="Manage models" onClick={() => setConfiguring(true)} /></>}
    view={configuring ? configurePanel : undefined}
    emptyText="No models added yet"
    placement={placement}
    panelClassName={configuring ? "w-max min-w-72" : undefined}
    onClose={onClose}
    />
  </div>
}
