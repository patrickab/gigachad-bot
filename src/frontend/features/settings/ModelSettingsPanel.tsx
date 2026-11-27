"use client"

import { ArrowLeft, Check, ChevronDown, ChevronRight, Plus, Trash2 } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { fetchPiCatalog } from "@/lib/backend/api"
import { displayName, PI_MODEL_SOURCE, PI_PROVIDER_LABEL } from "@/lib/models"
import type { ModelDefaults, ModelProvider, ModelsResponse, PiCatalog } from "@/lib/domain/types"
import { errorMessage, cn } from "@/lib/utils"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { StyledSelect } from "@/components/ui/StyledSelect"

type SettingsTab = "providers" | "defaults"

/** State of the model admin view, kept for the whole dropdown lifetime so the tab, draft defaults and Pi catalog survive closing it. */
export function useModelSettings(onProvidersChange: (providers: ModelProvider[]) => Promise<void>, onDefaultsChange: (defaults: ModelDefaults) => Promise<void>) {
  const [configuring, setConfiguring] = useState(false)
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("providers")
  const [error, setError] = useState<string | null>(null)
  const [draftDefaults, setDraftDefaults] = useState<ModelDefaults | null>(null)
  const [piCatalog, setPiCatalog] = useState<PiCatalog | null>(null)
  // Default-model edits are buffered and flushed once on close, not saved per keystroke.
  const pendingDefaults = useRef<ModelDefaults | null>(null)
  /** Leave the admin view and flush buffered default-model edits. */
  const close = useCallback(() => {
    setConfiguring(false)
    const defaults = pendingDefaults.current
    pendingDefaults.current = null
    if (defaults) void onDefaultsChange(defaults).catch((reason) => setError(errorMessage(reason, "Could not save default models")))
  }, [onDefaultsChange])
  useEscapeLayer(() => setConfiguring(false), configuring)
  // Loaded when the settings panel opens, not on mount: pi-host is optional, so
  // probing its gateway for every chat view would cost a request nobody asked
  // for. A failed probe stays null and simply hides the section.
  useEffect(() => {
    if (!configuring || piCatalog) return
    fetchPiCatalog().then(setPiCatalog).catch(() => undefined)
  }, [configuring, piCatalog])

  const persist = async (next: ModelProvider[]) => {
    setError(null)
    try { await onProvidersChange(next); return true } catch (reason) { setError(errorMessage(reason, "Could not save providers")); return false }
  }
  const updateDefaults = (defaults: ModelDefaults) => { setDraftDefaults(defaults); pendingDefaults.current = defaults }

  return { configuring, setConfiguring, settingsTab, setSettingsTab, error, setError, draftDefaults, piCatalog, persist, updateDefaults, close }
}

/** Admin view state shared by the dropdown (open/close, error line) and the panel. */
export type ModelSettings = ReturnType<typeof useModelSettings>

/** Provider/model admin view of the model dropdown: providers, Pi picks and default models. */
export function ModelSettingsPanel({ settings, models, choices }: { settings: ModelSettings; models: ModelsResponse; choices: string[] }) {
  const segmentClass = (active: boolean) => cn("flex-1 rounded-md px-2 py-1.5 text-xs transition-colors", active ? "bg-surface-elevated text-ink shadow-[var(--shadow-sm)]" : "text-ink-subtle hover:text-ink")
  return <>
    <div className="border-b border-divider p-1.5">
      <div className="flex rounded-lg bg-paper p-0.5">
        <button aria-label="Back to model selector" onClick={() => settings.setConfiguring(false)} className="rounded-md px-2 py-1.5 text-ink-subtle hover:bg-surface-elevated hover:text-ink"><ArrowLeft className="h-3.5 w-3.5" /></button>
        <button onClick={() => settings.setSettingsTab("providers")} className={segmentClass(settings.settingsTab === "providers")}>Providers / Models</button>
        <button onClick={() => settings.setSettingsTab("defaults")} className={segmentClass(settings.settingsTab === "defaults")}>Default Models</button>
      </div>
    </div>
    {settings.settingsTab === "providers"
      ? <ProviderSettings providers={models.providers} piCatalog={settings.piCatalog} onChange={settings.persist} />
      : <DefaultModels defaults={settings.draftDefaults ?? models.defaults} choices={choices} onChange={settings.updateDefaults} />}
    <ModelSettingsError error={settings.error} />
  </>
}

/** Save error line shown under the selector and the admin view. */
export function ModelSettingsError({ error }: { error: string | null }) {
  return error && <p className="px-2 pb-1.5 text-xs text-danger">{error}</p>
}

function AddProvider({ onAdd }: { onAdd: (label: string, providerId: string) => void }) {
  const [label, setLabel] = useState("")
  const [providerId, setProviderId] = useState("")
  const submit = () => onAdd(label, providerId)

  return <div className="grid grid-cols-[1fr_1fr_auto] gap-1 border-t border-divider pt-2">
    <input autoFocus value={label} onChange={(event) => setLabel(event.target.value)} placeholder="Label" className="min-w-0 rounded border border-divider bg-paper px-2 py-1.5 text-xs text-ink outline-none" />
    <input value={providerId} onChange={(event) => setProviderId(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") submit() }} placeholder="Provider ID" className="min-w-0 rounded border border-divider bg-paper px-2 py-1.5 text-xs text-ink outline-none" />
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

/** `providers` with the models of the provider labelled `label` replaced by `models(current)`. */
const patchProvider = (providers: ModelProvider[], label: string, models: (current: string[]) => string[]): ModelProvider[] =>
  providers.map((item) => item.label === label ? { ...item, models: models(item.models) } : item)

/** Picks models out of Pi's own logins, including OAuth subscription plans.
 *
 * Every pick lands in the same provider catalog as a hand-entered provider:
 * one `Pi` row prefixed `litellm_proxy`, which the backend routes to pi-host's
 * gateway, and the selector shows a single Pi tab instead of a tab per login.
 * Nothing downstream of the catalog needs to know Pi exists. The picker still
 * browses by login, because that is how Pi holds credentials; only the
 * destination row is shared.
 */
function PiSection({ catalog, providers, onChange }: {
  catalog: PiCatalog | null
  providers: ModelProvider[]
  onChange: (next: ModelProvider[]) => Promise<boolean>
}) {
  const [expanded, setExpanded] = useState<string | null>(null)
  if (!catalog?.installed) return null
  if (!catalog.online) return <p className="border-t border-divider px-3 py-2 text-[10px] text-ink-subtle">pi-host is set up but its gateway is not answering on {catalog.gateway}. Start it with run-pi-host.sh to add subscription models.</p>

  const piModels = providers.find((item) => item.label === PI_PROVIDER_LABEL)?.models ?? []
  const toggle = (modelId: string) => {
    if (!providers.some((item) => item.label === PI_PROVIDER_LABEL)) return onChange([...providers, { label: PI_PROVIDER_LABEL, provider_id: catalog.provider_id, source: PI_MODEL_SOURCE, models: [modelId] }])
    return onChange(patchProvider(providers, PI_PROVIDER_LABEL, (models) => models.includes(modelId) ? models.filter((value) => value !== modelId) : [...models, modelId]))
  }

  return <div className="border-t border-divider">
    <p className="px-3 pb-1 pt-2 text-[10px] uppercase tracking-wide text-ink-subtle">From Pi</p>
    <div className="max-h-64 space-y-0.5 overflow-y-auto p-1.5 pt-0">
      {catalog.providers.map((provider) => {
        const picked = provider.models.filter((model) => piModels.includes(model.id))
        const open = expanded === provider.id
        return <div key={provider.id} className="rounded-md hover:bg-surface-elevated">
          <button onClick={() => setExpanded(open ? null : provider.id)} className="flex w-full items-center">
            <span className="p-2 text-ink-subtle">{open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}</span>
            <span className="flex min-w-0 flex-1 items-baseline gap-2 py-2 text-left"><span className="truncate text-xs text-ink">{provider.label}</span><span className="shrink-0 text-[10px] text-ink-subtle">{picked.length} of {provider.models.length} added</span></span>
          </button>
          {open && <div className="border-t border-divider px-2 pb-2 pt-1">
            {provider.models.map((model) => {
              const added = piModels.includes(model.id)
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

function ProviderSettings({ providers, piCatalog, onChange }: { providers: ModelProvider[]; piCatalog: PiCatalog | null; onChange: (next: ModelProvider[]) => Promise<boolean> }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const addProvider = async (labelInput: string, providerIdInput: string) => {
    const label = labelInput.trim(), provider_id = providerIdInput.trim()
    if (!label || !provider_id || providers.some((provider) => provider.label.toLowerCase() === label.toLowerCase())) return
    if (await onChange([...providers, { label, provider_id, models: [] }])) setAdding(false)
  }
  return <>
    <div className="space-y-0.5 p-1.5">
      {providers.map((provider) => <ProviderRow key={provider.label} provider={provider} expanded={expanded === provider.label} onToggle={() => setExpanded(expanded === provider.label ? null : provider.label)} onAddModel={(model) => { const value = model.trim(); if (value && !provider.models.includes(value)) void onChange(patchProvider(providers, provider.label, (current) => [...current, value])) }} onDeleteModel={(model) => void onChange(patchProvider(providers, provider.label, (current) => current.filter((value) => value !== model)))} onDeleteProvider={() => void onChange(providers.filter((item) => item.label !== provider.label))} />)}
      {adding ? <AddProvider onAdd={(label, id) => void addProvider(label, id)} /> : <button onClick={() => setAdding(true)} className="flex w-full items-center gap-1 rounded-md px-2 py-2 text-xs text-ink-muted hover:bg-surface-elevated hover:text-ink"><Plus className="h-3.5 w-3.5" /> Add provider</button>}
    </div>
    <PiSection catalog={piCatalog} providers={providers} onChange={onChange} />
  </>
}

function DefaultModels({ defaults, choices, onChange }: { defaults: ModelDefaults; choices: string[]; onChange: (next: ModelDefaults) => void }) {
  const labels: Record<keyof ModelDefaults, string> = { default_model: "Chat", small_model: "Small / fast", vision_model: "Vision / OCR", sandbox_model: "Sandbox" }
  const options = choices.map((model) => ({ value: model, label: model }))
  return <div className="space-y-2 p-2">{(Object.keys(labels) as (keyof ModelDefaults)[]).map((key) => <label key={key} className="block"><span className="mb-1 block px-1 text-[10px] text-ink-subtle">{labels[key]}</span><StyledSelect ariaLabel={`${labels[key]} model`} value={defaults[key]} options={options} onChange={(model) => void onChange({ ...defaults, [key]: model })} /></label>)}</div>
}
