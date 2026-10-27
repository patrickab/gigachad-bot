"use client"

import { useEffect, useState } from "react"
import { Settings } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { errorMessage, cn } from "@/lib/utils"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { fetchFactoryDefaults, fetchModels, saveFactoryDefaults } from "@/lib/backend/api"
import { FACTORY_AGENT_NAMES, type FactoryAgentName, type FactoryDefaults, type FactorySettings, type MergeMode, type ModelsResponse } from "@/lib/domain/types"
import { Field } from "./Field"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"

const MERGE_OPTIONS = [
  { value: "review", label: "Review" },
  { value: "auto", label: "Auto" },
]

const MAX_ROUNDS_LIMITS = { min: 1, max: 20 }

/** Models the user filled in; an empty selector means the agent keeps its own model. */
function withoutEmptyModels(settings: FactorySettings): FactorySettings {
  return { ...settings, models: Object.fromEntries(Object.entries(settings.models).filter(([, selector]) => selector.trim() !== "")) }
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-sm text-ink">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} className="h-3.5 w-3.5 accent-ink" />
      {label}
    </label>
  )
}

/** Modal that edits the engine-wide factory defaults: workflow settings, a model per agent and each agent's prompt file. */
export function FactorySettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [draft, setDraft] = useState<FactoryDefaults | null>(null)
  const [catalog, setCatalog] = useState<ModelsResponse | null>(null)
  const [modelError, setModelError] = useState("")
  const [agent, setAgent] = useState<FactoryAgentName>(FACTORY_AGENT_NAMES[0])
  const { run: persist, pending: busy, error, setError } = useAsyncAction(async () => {
    if (!draft) return
    await saveFactoryDefaults({ ...draft, settings: withoutEmptyModels(draft.settings) })
    onClose()
  }, "Could not save the defaults")

  // Every opening reloads the saved defaults, so a discarded edit never lingers.
  useResetOnOpen(open, () => {
    setDraft(null)
    setCatalog(null)
    setModelError("")
    setAgent(FACTORY_AGENT_NAMES[0])
    setError("")
  })

  useEffect(() => {
    if (!open) return
    let cancelled = false
    fetchFactoryDefaults().then((loaded) => {
      if (!cancelled) setDraft(loaded)
    }).catch((cause: unknown) => {
      if (!cancelled) setError(errorMessage(cause, "Could not load the defaults"))
    })
    fetchModels().then((loaded) => {
      if (!cancelled) setCatalog(loaded)
    }).catch((cause: unknown) => {
      if (!cancelled) setModelError(errorMessage(cause, "Could not load OMP models"))
    })
    return () => { cancelled = true }
  }, [open, setError])

  const changeSettings = (change: Partial<FactorySettings>) =>
    setDraft((current) => current && { ...current, settings: { ...current.settings, ...change } })
  const changeModel = (name: string, selector: string) =>
    setDraft((current) => current && { ...current, settings: { ...current.settings, models: { ...current.settings.models, [name]: selector } } })
  const changeAgent = (markdown: string) =>
    setDraft((current) => current && { ...current, agents: { ...current.agents, [agent]: markdown } })

  const modelOptions = [
    { value: "", label: "Agent's own model" },
    ...(catalog?.providers.filter((provider) => provider.source === "omp").flatMap((provider) => provider.models.map((model) => ({
      value: model,
      label: model,
    }))) ?? []),
  ]

  const save = () => {
    if (draft && !busy) void persist()
  }

  return (
    <Dialog open={open} onClose={onClose} icon={Settings} title="Factory defaults" widthClassName="max-w-2xl">
      <form onSubmit={(event) => { event.preventDefault(); void save() }} className="flex flex-col gap-4 px-5 pb-5">
        <p className="-mt-1 text-[11px] text-ink-faint">Repos override these with factory.yaml and .omp/agents/</p>
        {draft ? (
          <div className="flex max-h-[60vh] flex-col gap-4 overflow-y-auto pr-1 scrollbar-thin">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Merge">
                <StyledSelect ariaLabel="Merge" value={draft.settings.merge} onChange={(value) => changeSettings({ merge: value as MergeMode })} options={MERGE_OPTIONS} />
              </Field>
              <Field label="Max rounds">
                <input
                  type="number"
                  aria-label="Max rounds"
                  min={MAX_ROUNDS_LIMITS.min}
                  max={MAX_ROUNDS_LIMITS.max}
                  value={Number.isNaN(draft.settings.max_rounds) ? "" : draft.settings.max_rounds}
                  onChange={(event) => changeSettings({ max_rounds: event.target.valueAsNumber })}
                  className={FIELD_CLASS}
                />
              </Field>
            </div>
            <div className="flex gap-6">
              <Toggle label="Sandbox" checked={draft.settings.sandbox} onChange={(sandbox) => changeSettings({ sandbox })} />
              <Toggle label="Preview" checked={draft.settings.preview} onChange={(preview) => changeSettings({ preview })} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              {FACTORY_AGENT_NAMES.map((name) => {
                const selected = draft.settings.models[name] ?? ""
                const options = modelOptions.some((option) => option.value === selected)
                  ? modelOptions
                  : [...modelOptions, { value: selected, label: `${selected} (saved, not added)` }]
                return (
                  <Field key={name} label={`Model of ${name}`}>
                    <StyledSelect
                      ariaLabel={`Model of ${name}`}
                      value={selected}
                      onChange={(value) => changeModel(name, value)}
                      options={options}
                    />
                  </Field>
                )
              })}
            </div>
            {modelError
              ? <ErrorText>{modelError}</ErrorText>
              : <p role="status" className="text-[11px] text-ink-faint">{catalog ? "OMP models added in Providers / Models. Applies to new tasks." : "Loading OMP models"}</p>}
            <div className="flex flex-col gap-2">
              <div role="tablist" aria-label="Agent" className="flex gap-1">
                {FACTORY_AGENT_NAMES.map((name) => (
                  <button
                    key={name}
                    id={`agent-tab-${name}`}
                    type="button"
                    role="tab"
                    aria-selected={agent === name}
                    aria-controls="agent-markdown"
                    onClick={() => setAgent(name)}
                    className={cn(
                      "rounded-full px-3 py-1 text-[11px] font-medium transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted",
                      agent === name ? "bg-surface-elevated text-ink" : "text-ink-muted hover:bg-surface/50 hover:text-ink",
                    )}
                  >
                    {name}
                  </button>
                ))}
              </div>
              <textarea
                id="agent-markdown"
                role="tabpanel"
                aria-labelledby={`agent-tab-${agent}`}
                aria-label={`Markdown of ${agent}`}
                value={draft.agents[agent] ?? ""}
                onChange={(event) => changeAgent(event.target.value)}
                rows={14}
                className={cn(FIELD_CLASS, "resize-none font-mono text-xs leading-relaxed")}
                spellCheck={false}
              />
            </div>
          </div>
        ) : (
          !error && <div className="py-8 text-center text-xs text-ink-faint">Loading defaults</div>
        )}
        <ErrorText>{error}</ErrorText>
        <div className="flex items-center justify-end gap-2">
          <button type="button" onClick={onClose} className={QUIET_BUTTON_CLASS}>Cancel</button>
          <button type="submit" disabled={!draft || busy} className={PRIMARY_BUTTON_CLASS}>Save</button>
        </div>
      </form>
    </Dialog>
  )
}
