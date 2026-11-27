"use client"

import { useCallback, useState } from "react"
import { Settings, X } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { PillTabs } from "@/components/ui/PillTabs"
import { Switch } from "@/components/ui/Switch"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { IconButton } from "@/components/ui/IconButton"
import { errorMessage } from "@/lib/utils"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { callFactory, fetchModels, getFactoryActions } from "@/lib/backend/api"
import type { ModelsResponse } from "@/lib/domain/types"
import { gatewayModelId, qualifiedModelIds } from "@/lib/models"
import type { FactoryActions, FactoryBranch, FactoryMergeMode, FactoryModelList, FactoryModels, FactoryParamSchema, FactoryRepoSettings, FactoryStoredSettings } from "@/lib/domain/factory"
import { Field } from "@/components/ui/Field"
import { COMPACT_FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"

/** The merge mode as pills: merge on its own, or wait for the user's approve. */
export const MERGE_OPTIONS = [{ value: "auto", label: "Auto" }, { value: "review", label: "Approve" }] as const

/** The two models a task override sets: the big model (the coder) and the reviewer. */
export const AGENTS = ["big", "reviewer"] as const satisfies readonly (keyof FactoryModels)[]

/** The project's two model sizes: easy tasks of a wave run on small, everything else and the reviewer on big. */
export const MODEL_SIZES = [{ key: "small", label: "Small model" }, { key: "big", label: "Big model" }] as const

/** The sizes the dialog edits, "" for none. */
type SizeModels = Record<(typeof MODEL_SIZES)[number]["key"], string>

/** The sizes of stored `models`. */
export function sizeModelsOf(models: FactoryModels): SizeModels {
  return { small: models.small ?? "", big: models.big ?? "" }
}

/** The stored `models` of the sizes: the reviewer runs on big, an empty size is left out (Pi's default). */
export function storedModelsOf({ small, big }: SizeModels): FactoryModels {
  return { ...(small.trim() ? { small } : {}), ...(big.trim() ? { big, reviewer: big } : {}) }
}

/** Every model of the chat model selector the gateway serves, sorted, as the `<provider>/<id>` Pi's registry resolves: what the factory models list can add. */
export function factoryModelIds(catalog: ModelsResponse | null): string[] {
  if (!catalog) return []
  const unavailable = new Set(catalog.unavailable)
  const selectors = [...catalog.ollama, ...catalog.providers.flatMap(qualifiedModelIds)]
  return [...new Set(selectors.filter((selector) => !unavailable.has(selector)).map(gatewayModelId))].sort((a, b) => a.localeCompare(b))
}

/** The user's factory models through the core: the only models a task may use. */
export async function loadFactoryModels(): Promise<string[]> {
  return (await callFactory<FactoryModelList>("models.get", {})).models
}

/** Where a task would start while the factory models list is empty: the core refuses it. */
export const NO_FACTORY_MODELS = "No factory models are set (add them in the factory settings)"

/**
 * Options of an agent's model select: `fallback` for no model (the value ""), then the factory models list. A `selected`
 * model that is not on the list stays selectable, marked.
 */
export function agentModelOptions(fallback: string, models: readonly string[], selected: string): { value: string; label: string }[] {
  const options = [{ value: "", label: fallback }, ...models.map((model) => ({ value: model, label: model }))]
  if (selected === "" || models.includes(selected)) return options
  return [...options, { value: selected, label: `${selected} (not in the list)` }]
}

/** A branch the user filled in, null for an empty field: the default branch. */
export function branchOrDefault(branch: string | null): string | null {
  return branch?.trim() || null
}

/** The repo's settings through the core. */
export function loadRepoSettings(repo: string): Promise<FactoryRepoSettings> {
  return callFactory<FactoryRepoSettings>("settings.get", { repo })
}

/** The names of the repo's local branches through the core. */
export async function loadBranchNames(repo: string): Promise<string[]> {
  const { branches } = await callFactory<{ branches: FactoryBranch[] }>("branches.list", { repo })
  return branches.map((branch) => branch.name)
}

/**
 * Options of a branch select: `fallback` for no branch (the value ""), then `branches`. A `value` that is not among
 * the loaded branches stays selectable, marked missing; while they load (null) it is shown as it is.
 */
export function branchOptions(fallback: string, branches: string[] | null, value: string): { value: string; label: string }[] {
  const options = [{ value: "", label: fallback }, ...(branches ?? []).map((branch) => ({ value: branch, label: branch }))]
  if (value === "" || branches?.includes(value)) return options
  return [...options, { value, label: branches ? `${value} (missing)` : value }]
}

/** Labelled select of a branch of the repo, "" for the fallback. */
export function BranchSelect({ label, value, fallback, branches, disabled, onChange }: {
  label: string
  value: string
  fallback: string
  branches: string[] | null
  disabled?: boolean
  onChange: (value: string) => void
}) {
  return (
    <Field label={label}>
      <StyledSelect quiet ariaLabel={label} value={value} options={branchOptions(fallback, branches, value)} disabled={disabled} onChange={onChange} />
    </Field>
  )
}

/** The bounds of `maxRounds` the core's `settings.set` declares, none while unknown. */
export function maxRoundsLimits(actions: FactoryActions | null): { min?: number; max?: number } {
  const schema: FactoryParamSchema | undefined = actions?.actions["settings.set"]?.params.properties.settings?.properties?.maxRounds
  return { min: schema?.minimum, max: schema?.maximum }
}

/** What the dialog edits: every setting at its resolved value, the models and branch only as stored ("" for none). */
interface SettingsDraft {
  merge: FactoryMergeMode
  maxRounds: number
  review: boolean
  preview: boolean
  models: SizeModels
  branch: string
  /** The branch an unset one falls back to (the repo's default). */
  defaultBranch: string
  /** The settings as loaded: what the file holds and what they resolve to. */
  loaded: FactoryRepoSettings
}

function draftOf({ stored, resolved, defaults }: FactoryRepoSettings): SettingsDraft {
  return {
    merge: resolved.merge,
    maxRounds: resolved.maxRounds,
    review: resolved.review,
    preview: resolved.preview,
    models: sizeModelsOf(stored.models ?? {}),
    branch: stored.branch ?? "",
    defaultBranch: defaults.branch,
    loaded: { stored, resolved, defaults },
  }
}

/** Keys the dialog shows at their resolved value. */
const RESOLVED_KEYS = ["merge", "maxRounds", "review", "preview"] as const

/**
 * The settings file `draft` saves: the stored settings the dialog does not show (`squash`) as they are, a setting the file held or the user changed (one left at its default stays unset,
 * so it follows the defaults), models the user filled in (an empty one runs Pi's default), a branch only when picked.
 */
function storedOf(draft: SettingsDraft): FactoryStoredSettings {
  const { stored, resolved } = draft.loaded
  const branch = branchOrDefault(draft.branch)
  const kept = RESOLVED_KEYS.filter((key) => stored[key] !== undefined || draft[key] !== resolved[key])
  return {
    ...Object.fromEntries(Object.entries(stored).filter(([key]) => key !== "branch")),
    ...Object.fromEntries(kept.map((key) => [key, draft[key]])),
    models: storedModelsOf(draft.models),
    ...(branch ? { branch } : {}),
  }
}

function validRounds(rounds: number, { min, max }: { min?: number; max?: number }): boolean {
  return Number.isInteger(rounds) && (min === undefined || rounds >= min) && (max === undefined || rounds <= max)
}

/** The label of a choice row: one width, so the pills and switches of a column line up. */
const CHOICE_LABEL_CLASS = "w-20 shrink-0 text-[11px] text-ink-subtle"

/** A labelled row of pills of the factory dialogs, one of `options` selected. */
export function PillChoice<T extends string>({ label, options, value, disabled, onChange }: {
  label: string
  options: readonly { value: T; label: string }[]
  value: T
  disabled?: boolean
  onChange: (value: T) => void
}) {
  return (
    <div role="group" aria-label={label} className="flex h-7 items-center">
      <span className={CHOICE_LABEL_CLASS}>{label}</span>
      <PillTabs quiet options={options} value={value} onChange={onChange} disabled={disabled} />
    </div>
  )
}

/** A labelled on/off choice of the factory dialogs, the app's `Switch`. */
export function SwitchChoice({ label, checked, disabled, onChange }: { label: string; checked: boolean; disabled?: boolean; onChange: (checked: boolean) => void }) {
  return (
    <div className="flex h-7 items-center">
      <span aria-hidden className={CHOICE_LABEL_CLASS}>{label}</span>
      <Switch label={label} checked={checked} disabled={disabled} onToggle={() => onChange(!checked)} />
    </div>
  )
}

/** The factory models list: each model removable, more added from the models gigachad knows (`known`). */
function FactoryModelsField({ models, known, onChange }: { models: readonly string[]; known: readonly string[]; onChange: (models: string[]) => void }) {
  const addable = known.filter((model) => !models.includes(model))
  return (
    <Field label="Models">
      {models.length > 0 && (
        <ul aria-label="Factory models" className="flex flex-col">
          {models.map((model) => (
            <li key={model} className="group flex h-7 items-center gap-2 pl-2.5 text-xs text-ink-muted">
              <span className="min-w-0 flex-1 truncate">{model}</span>
              <IconButton
                icon={X}
                label={`Remove ${model}`}
                onClick={() => onChange(models.filter((kept) => kept !== model))}
                className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
              />
            </li>
          ))}
        </ul>
      )}
      <StyledSelect
        quiet
        ariaLabel="Add model"
        value=""
        options={[{ value: "", label: "Add model" }, ...addable.map((model) => ({ value: model, label: model }))]}
        onChange={(model) => { if (model) onChange([...models, model]) }}
      />
    </Field>
  )
}

/**
 * Modal that edits the factory settings of the repo at path `repo` (merge, rounds, branch, review, preview and a model
 * per agent) and the user's factory models list, which every repo shares: Save stores a changed list before the settings.
 */
export function FactorySettingsDialog({ repo, open, onClose }: { repo: string; open: boolean; onClose: () => void }) {
  const [draft, setDraft] = useState<SettingsDraft | null>(null)
  const [catalog, setCatalog] = useState<ModelsResponse | null>(null)
  const [modelError, setModelError] = useState("")
  /** The factory models list as edited and as the core holds it, null until loaded. */
  const [factoryModels, setFactoryModels] = useState<string[] | null>(null)
  const [savedFactoryModels, setSavedFactoryModels] = useState<string[] | null>(null)
  const [factoryModelsError, setFactoryModelsError] = useState("")
  const [branches, setBranches] = useState<string[] | null>(null)
  const [branchError, setBranchError] = useState("")
  const [actions, setActions] = useState<FactoryActions | null>(null)
  const { run: persist, pending: busy, error, setError } = useAsyncAction(async () => {
    if (!draft) return
    if (factoryModels !== null && factoryModels.join("\n") !== savedFactoryModels?.join("\n")) {
      const saved = await callFactory<FactoryModelList>("models.set", { models: factoryModels })
      setSavedFactoryModels(saved.models)
    }
    await callFactory<FactoryRepoSettings>("settings.set", { repo, settings: storedOf(draft) })
    onClose()
  }, "Could not save the settings")

  // Every opening reloads the saved settings, so a discarded edit never lingers.
  useResetOnOpen(open, () => {
    setDraft(null)
    setCatalog(null)
    setModelError("")
    setFactoryModels(null)
    setSavedFactoryModels(null)
    setFactoryModelsError("")
    setBranches(null)
    setBranchError("")
    setError("")
  })

  const loadSettings = useCallback(async () => draftOf(await loadRepoSettings(repo)), [repo])
  useLoadWhile(open, loadSettings, setDraft, (cause) => setError(errorMessage(cause, "Could not load the settings")))
  useLoadWhile(open, fetchModels, setCatalog, (cause) => setModelError(errorMessage(cause, "Could not load Pi models")))
  useLoadWhile(open, loadFactoryModels, (models) => {
    setFactoryModels(models)
    setSavedFactoryModels(models)
  }, (cause) => setFactoryModelsError(errorMessage(cause, "Could not load the factory models")))
  useLoadWhile(open, getFactoryActions, setActions)
  const roundsLimits = maxRoundsLimits(actions)
  const loadBranches = useCallback(() => loadBranchNames(repo), [repo])
  useLoadWhile(open, loadBranches, setBranches, (cause) => setBranchError(errorMessage(cause, "Could not load the branches")))

  const change = (changed: Partial<SettingsDraft>) => setDraft((current) => current && { ...current, ...changed })
  const changeModel = (size: keyof SizeModels, selector: string) =>
    setDraft((current) => current && { ...current, models: { ...current.models, [size]: selector } })

  const canSave = draft !== null && !busy && validRounds(draft.maxRounds, roundsLimits)
  const save = () => {
    if (canSave) void persist()
  }

  return (
    <Dialog open={open} onClose={onClose} icon={Settings} title="Factory settings" widthClassName="max-w-lg">
      <form onSubmit={(event) => { event.preventDefault(); save() }} className="flex flex-col gap-4 px-5 pb-5">
        {draft ? (
          <>
            <div className="grid grid-cols-2 gap-3">
              {MODEL_SIZES.map(({ key, label }) => {
                const selected = draft.models[key]
                return (
                  <Field key={key} label={label}>
                    <StyledSelect
                      quiet
                      ariaLabel={label}
                      value={selected}
                      onChange={(value) => changeModel(key, value)}
                      options={agentModelOptions("Pi default", factoryModels ?? [], selected)}
                    />
                  </Field>
                )
              })}
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-2">
              <PillChoice label="Merge" options={MERGE_OPTIONS} value={draft.merge} onChange={(merge) => change({ merge })} />
              <SwitchChoice label="Review" checked={draft.review} onChange={(review) => change({ review })} />
              <SwitchChoice label="Preview" checked={draft.preview} onChange={(preview) => change({ preview })} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <BranchSelect
                label="Branch"
                value={draft.branch}
                fallback={`Default (${draft.defaultBranch})`}
                branches={branches}
                onChange={(branch) => change({ branch })}
              />
              <Field label="Max rounds">
                <input
                  type="number"
                  aria-label="Max rounds"
                  min={roundsLimits.min}
                  max={roundsLimits.max}
                  value={Number.isNaN(draft.maxRounds) ? "" : draft.maxRounds}
                  onChange={(event) => change({ maxRounds: event.target.valueAsNumber })}
                  className={COMPACT_FIELD_CLASS}
                />
              </Field>
            </div>
            {factoryModels && <FactoryModelsField models={factoryModels} known={factoryModelIds(catalog)} onChange={setFactoryModels} />}
            <ErrorText>{branchError}</ErrorText>
            <ErrorText>{factoryModelsError}</ErrorText>
            <ErrorText>{modelError}</ErrorText>
          </>
        ) : (
          !error && <div role="status" className="py-8 text-center text-xs text-ink-faint">Loading settings</div>
        )}
        <ErrorText>{error}</ErrorText>
        <div className="flex items-center justify-end gap-2">
          <button type="button" onClick={onClose} className={QUIET_BUTTON_CLASS}>Cancel</button>
          <button type="submit" disabled={!canSave} className={PRIMARY_BUTTON_CLASS}>Save</button>
        </div>
      </form>
    </Dialog>
  )
}
