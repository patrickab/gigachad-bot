"use client"

import { AnimatePresence, motion, Reorder } from "framer-motion"
import { ArrowLeft, Check, ChevronDown, ChevronRight, MoreHorizontal, Plus, Trash2 } from "lucide-react"
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { useClickOutside } from "@/hooks/useClickOutside"
import { fetchOmpCatalog } from "@/lib/api"
import { displayName } from "@/lib/models"
import type { ModelDefaults, ModelProvider, ModelsResponse, OmpCatalog } from "@/lib/types"
import { cn } from "@/lib/utils"
import { Skeleton } from "./Skeleton"
import { StyledSelect } from "./StyledSelect"

interface Tab { key: string; label: string }
interface ProviderTab { label: string; models: string[] }
type Panel = "models" | "configure" | null

interface ModelDropdownProps {
  models: ModelsResponse | null
  selectedModel: string
  onSelect: (model: string) => void
  onProvidersChange: (providers: ModelProvider[]) => Promise<void>
  onTabOrderChange?: (order: string[]) => Promise<void>
  onDefaultsChange: (defaults: ModelDefaults) => Promise<void>
  accent?: "sky" | "amber"
  extraTabs?: Tab[]
  activeExtraTab?: string
  onExtraTabChange?: (key: string) => void
  children?: (props: { open: boolean; onToggle: () => void }) => ReactNode
}

const ACCENT_CLASSES = {
  sky: { selected: "bg-surface-elevated text-ink", check: "h-4 w-4" },
  amber: { selected: "bg-surface-elevated text-ink", check: "h-3.5 w-3.5" },
} as const

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
      <button onClick={onToggle} className="min-w-0 flex-1 py-2 text-left"><p className="truncate text-xs text-ink">{provider.label}</p><p className="truncate text-[10px] text-ink-subtle">{provider.source ? `${provider.source} · ` : ""}{provider.litellm_id}/ · {provider.models.length} models</p></button>
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
 * Every pick lands in the same provider catalog as a hand-entered provider —
 * one `OMP` row prefixed `litellm_proxy` so LiteLLM routes it at the OMP
 * gateway — so nothing downstream of the catalog needs to know OMP exists.
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
            <span className="min-w-0 flex-1 py-2 text-left"><span className="block truncate text-xs text-ink">{provider.label}</span><span className="block truncate text-[10px] text-ink-subtle">{picked.length} of {provider.models.length} added</span></span>
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
  const labels: Record<keyof ModelDefaults, string> = { default_model: "Chat", small_model: "Small / fast", vision_model: "Vision / OCR", memory_model: "Memory" }
  const options = choices.map((model) => ({ value: model, label: model }))
  return <div className="space-y-2 p-2">{(Object.keys(labels) as (keyof ModelDefaults)[]).map((key) => <label key={key} className="block"><span className="mb-1 block px-1 text-[10px] text-ink-subtle">{labels[key]}</span><StyledSelect ariaLabel={`${labels[key]} model`} value={defaults[key]} options={options} onChange={(model) => void onChange({ ...defaults, [key]: model })} /></label>)}</div>
}

export function ModelDropdown({ models, selectedModel, onSelect, onProvidersChange, onTabOrderChange, onDefaultsChange, accent = "sky", extraTabs, activeExtraTab, onExtraTabChange, children }: ModelDropdownProps) {
  const [panel, setPanel] = useState<Panel>(null)
  const [settingsTab, setSettingsTab] = useState<"providers" | "defaults">("providers")
  const [manualTab, setManualTab] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [draftDefaults, setDraftDefaults] = useState<ModelDefaults | null>(null)
  const [ompCatalog, setOmpCatalog] = useState<OmpCatalog | null>(null)
  const pendingDefaults = useRef<ModelDefaults | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const close = useCallback(() => {
    setPanel(null)
    const defaults = pendingDefaults.current
    pendingDefaults.current = null
    if (defaults) void onDefaultsChange(defaults).catch((reason) => setError((reason as Error).message || "Could not save default models"))
  }, [onDefaultsChange])
  useClickOutside(ref, close)
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape" && panel === "configure") setPanel("models") }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [panel])
  // Loaded when the settings panel opens, not on mount: OMP is optional, so
  // probing its gateway for every chat view would cost a request nobody asked
  // for. A failed probe stays null and simply hides the section.
  useEffect(() => {
    if (panel !== "configure" || ompCatalog) return
    fetchOmpCatalog().then(setOmpCatalog).catch(() => undefined)
  }, [panel, ompCatalog])

  if (models === null) return <div className="flex flex-col gap-1.5 px-2"><Skeleton className="h-5 w-16" /><Skeleton className="h-3 w-28" /></div>

  const available: ProviderTab[] = [
    { label: "Ollama", models: models.ollama },
    ...models.providers.map((provider) => ({ label: provider.label, models: provider.models.map((model) => `${provider.litellm_id}/${model}`) })),
  ]
  const selectableProviders = available.filter((provider) => provider.models.length > 0)
  const tabOrder = models.tab_order
  // Explicit position from the saved tab order; unlisted tabs (never dragged,
  // or new since the order was saved) fall to the end, alphabetical — same
  // rank rule PromptStore uses for prompt display order.
  const orderedTabs = [...selectableProviders].sort((a, b) => {
    const ai = tabOrder.indexOf(a.label)
    const bi = tabOrder.indexOf(b.label)
    if (ai !== -1 && bi !== -1) return ai - bi
    if (ai !== -1) return -1
    if (bi !== -1) return 1
    return a.label.localeCompare(b.label)
  })
  const choices = available.flatMap((provider) => provider.models)
  // Opens on whichever provider holds the current selection — falling back
  // to a manual tab click, then the first tab — so a configured default
  // model (e.g. an OMP pick) doesn't get stranded behind the Ollama tab.
  const activeLabel = manualTab ?? orderedTabs.find((provider) => provider.models.includes(selectedModel))?.label ?? orderedTabs[0]?.label
  const active = orderedTabs.find((provider) => provider.label === activeLabel) ?? orderedTabs[0]
  const persist = async (next: ModelProvider[]) => {
    setError(null)
    try { await onProvidersChange(next); return true } catch (reason) { setError((reason as Error).message || "Could not save providers"); return false }
  }
  const updateDefaults = (defaults: ModelDefaults) => { setDraftDefaults(defaults); pendingDefaults.current = defaults }
  const reorderTabs = (next: ProviderTab[]) => void onTabOrderChange?.(next.map((provider) => provider.label)).catch((reason) => setError((reason as Error).message || "Could not save tab order"))
  const trigger = <><button onClick={() => panel === "models" ? close() : setPanel("models")} className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-sm font-medium text-ink-muted hover:bg-surface hover:text-ink"><span>Model</span><ChevronDown className="h-3 w-3 text-ink-subtle" /></button>{selectedModel && <span className="max-w-[150px] truncate px-2 text-[10px] italic text-ink-subtle">{displayName(selectedModel)}</span>}</>

  return <div className="relative z-50 flex flex-col items-start" ref={ref}>
    {children ? children({ open: panel === "models", onToggle: () => panel === "models" ? close() : setPanel("models") }) : trigger}
    <AnimatePresence>{panel && <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.15 }} className={cn("absolute left-0 top-full mt-1 flex w-max min-w-72 flex-col rounded-xl border border-divider bg-surface shadow-[var(--shadow-lg)]", panel === "models" && "overflow-hidden")}>
      {panel === "configure" ? <><div className="border-b border-divider p-1.5"><div className="flex rounded-lg bg-paper p-0.5"><button aria-label="Back to model selector" onClick={() => setPanel("models")} className="rounded-md px-2 py-1.5 text-ink-subtle hover:bg-surface-elevated hover:text-ink"><ArrowLeft className="h-3.5 w-3.5" /></button><button onClick={() => setSettingsTab("providers")} className={cn("flex-1 rounded-md px-2 py-1.5 text-xs", settingsTab === "providers" ? "bg-surface-elevated text-ink shadow-[var(--shadow-sm)]" : "text-ink-subtle")}>Providers / Models</button><button onClick={() => setSettingsTab("defaults")} className={cn("flex-1 rounded-md px-2 py-1.5 text-xs", settingsTab === "defaults" ? "bg-surface-elevated text-ink shadow-[var(--shadow-sm)]" : "text-ink-subtle")}>Default Models</button></div></div>{settingsTab === "providers" ? <ProviderSettings providers={models.providers} ompCatalog={ompCatalog} onChange={persist} /> : <DefaultModels defaults={draftDefaults ?? models.defaults} choices={choices} onChange={updateDefaults} />}{error && <p className="px-2 pb-2 text-xs text-danger">{error}</p>}</> : <>
        <div className="shrink-0 space-y-1.5 border-b border-divider p-1.5">
          {extraTabs && extraTabs.length > 0 && <div className="flex rounded-lg bg-paper p-0.5">{extraTabs.map((tab) => <button key={tab.key} onClick={() => onExtraTabChange?.(tab.key)} className={cn("flex-1 rounded-md px-2 py-1.5 text-xs font-medium capitalize transition-colors", tab.key === activeExtraTab ? "bg-surface-elevated text-ink shadow-[var(--shadow-sm)]" : "text-ink-subtle hover:text-ink")}>{tab.label}</button>)}</div>}
          <div className="flex rounded-lg bg-paper p-0.5"><Reorder.Group as="div" axis="x" values={orderedTabs} onReorder={reorderTabs} className="flex flex-1">{orderedTabs.map((provider) => <Reorder.Item as="div" key={provider.label} value={provider} className="flex-1 cursor-grab active:cursor-grabbing"><button onClick={() => setManualTab(provider.label)} className={cn("w-full rounded-md px-2 py-1.5 text-xs font-medium transition-colors", provider.label === active?.label ? "bg-surface-elevated text-ink shadow-[var(--shadow-sm)]" : "text-ink-subtle hover:text-ink")}>{provider.label}</button></Reorder.Item>)}</Reorder.Group><button aria-label="Configure providers" onClick={() => setPanel("configure")} className="rounded-md px-1.5 text-ink-subtle hover:bg-surface-elevated hover:text-ink"><MoreHorizontal className="h-4 w-4" /></button></div>
        </div>
        <div className="space-y-0.5 p-1.5">{active?.models.map((model) => <button key={model} onClick={() => { onSelect(model); close() }} className={cn("flex w-full items-center justify-between rounded-md px-2 py-2 text-sm transition-colors", model === selectedModel ? ACCENT_CLASSES[accent].selected : "text-ink hover:bg-surface-elevated hover:text-ink")}><span className="truncate">{displayName(model)}</span>{model === selectedModel && <Check className={cn("shrink-0", ACCENT_CLASSES[accent].check)} />}</button>)}</div>
      </>}
    </motion.div>}</AnimatePresence>
  </div>
}
