"use client"

import { useCallback, useState } from "react"
import { Settings } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { errorMessage } from "@/lib/utils"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { fetchModels, getFactorySettings, listFactoryBranches, putFactorySettings } from "@/lib/backend/api"
import type { ModelsResponse } from "@/lib/domain/types"
import { gatewayModelId, qualifiedModelIds } from "@/lib/models"
import type { FactoryMergeMode, FactoryModels, FactorySettings } from "@/lib/domain/factory"
import { Field } from "@/components/ui/Field"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"

const MERGE_OPTIONS = [
  { value: "review", label: "Review" },
  { value: "auto", label: "Auto" },
]

const MAX_ROUNDS_LIMITS = { min: 1, max: 20 }

/** The two agents of a task, in pipeline order. */
const AGENTS = ["coder", "reviewer"] as const satisfies readonly (keyof FactoryModels)[]

/** A branch the user filled in, null for an empty field: the default branch. */
export function branchOrDefault(branch: string | null): string | null {
  return branch?.trim() || null
}

/** What an unset branch falls back to, named by the first option of its select. */
export const SOURCE_BRANCH_DEFAULT = "main, else master"
export const TARGET_BRANCH_DEFAULT = "same as source"

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
      <StyledSelect ariaLabel={label} value={value} options={branchOptions(fallback, branches, value)} disabled={disabled} onChange={onChange} />
    </Field>
  )
}

/** Models the user filled in; an empty selector means the agent runs on Pi's default model. An empty branch is the default one. */
function withoutEmpty(settings: FactorySettings): FactorySettings {
  return {
    ...settings,
    models: Object.fromEntries(Object.entries(settings.models).filter(([, selector]) => selector.trim() !== "")),
    source_branch: branchOrDefault(settings.source_branch),
    target_branch: branchOrDefault(settings.target_branch),
  }
}

function validRounds(rounds: number): boolean {
  return Number.isInteger(rounds) && rounds >= MAX_ROUNDS_LIMITS.min && rounds <= MAX_ROUNDS_LIMITS.max
}

/** Labelled checkbox of the factory dialogs. */
export function Toggle({ label, checked, disabled, onChange }: { label: string; checked: boolean; disabled?: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-sm text-ink">
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} className="h-3.5 w-3.5 accent-ink" />
      {label}
    </label>
  )
}

/** Modal that edits the factory settings of the repo at path `repo`: merge, rounds, source and target branch, review, worktree, preview and a model per agent. */
export function FactorySettingsDialog({ repo, open, onClose }: { repo: string; open: boolean; onClose: () => void }) {
  const [draft, setDraft] = useState<FactorySettings | null>(null)
  const [catalog, setCatalog] = useState<ModelsResponse | null>(null)
  const [modelError, setModelError] = useState("")
  const [branches, setBranches] = useState<string[] | null>(null)
  const [branchError, setBranchError] = useState("")
  const { run: persist, pending: busy, error, setError } = useAsyncAction(async () => {
    if (!draft) return
    await putFactorySettings(repo, withoutEmpty(draft))
    onClose()
  }, "Could not save the settings")

  // Every opening reloads the saved settings, so a discarded edit never lingers.
  useResetOnOpen(open, () => {
    setDraft(null)
    setCatalog(null)
    setModelError("")
    setBranches(null)
    setBranchError("")
    setError("")
  })

  const loadSettings = useCallback(() => getFactorySettings(repo), [repo])
  useLoadWhile(open, loadSettings, setDraft, (cause) => setError(errorMessage(cause, "Could not load the settings")))
  useLoadWhile(open, fetchModels, setCatalog, (cause) => setModelError(errorMessage(cause, "Could not load Pi models")))
  const loadBranches = useCallback(() => listFactoryBranches(repo), [repo])
  useLoadWhile(open, loadBranches, setBranches, (cause) => setBranchError(errorMessage(cause, "Could not load the branches")))

  const change = (changed: Partial<FactorySettings>) => setDraft((current) => current && { ...current, ...changed })
  const changeModel = (agent: keyof FactoryModels, selector: string) =>
    setDraft((current) => current && { ...current, models: { ...current.models, [agent]: selector } })

  // Every model of the chat model selector the gateway serves, sorted as it lists them, as the `<provider>/<id>` Pi's registry resolves.
  const unavailable = new Set(catalog?.unavailable)
  const selectors = catalog ? [...catalog.ollama, ...catalog.providers.flatMap(qualifiedModelIds)] : []
  const gatewayIds = [...new Set(selectors.filter((selector) => !unavailable.has(selector)).map(gatewayModelId))].sort((a, b) => a.localeCompare(b))
  const modelOptions = [
    { value: "", label: "Pi default" },
    ...gatewayIds.map((model) => ({ value: model, label: model })),
  ]

  const canSave = draft !== null && !busy && validRounds(draft.max_rounds)
  const save = () => {
    if (canSave) void persist()
  }

  return (
    <Dialog open={open} onClose={onClose} icon={Settings} title="Factory settings" widthClassName="max-w-lg">
      <form onSubmit={(event) => { event.preventDefault(); save() }} className="flex flex-col gap-4 px-5 pb-5">
        {draft ? (
          <>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Merge">
                <StyledSelect ariaLabel="Merge" value={draft.merge} onChange={(value) => change({ merge: value as FactoryMergeMode })} options={MERGE_OPTIONS} />
              </Field>
              <Field label="Max rounds">
                <input
                  type="number"
                  aria-label="Max rounds"
                  min={MAX_ROUNDS_LIMITS.min}
                  max={MAX_ROUNDS_LIMITS.max}
                  value={Number.isNaN(draft.max_rounds) ? "" : draft.max_rounds}
                  onChange={(event) => change({ max_rounds: event.target.valueAsNumber })}
                  className={FIELD_CLASS}
                />
              </Field>
              <BranchSelect
                label="Source branch"
                value={draft.source_branch ?? ""}
                fallback={`Default (${SOURCE_BRANCH_DEFAULT})`}
                branches={branches}
                onChange={(source_branch) => change({ source_branch })}
              />
              <BranchSelect
                label="Target branch"
                value={draft.target_branch ?? ""}
                fallback={`Default (${TARGET_BRANCH_DEFAULT})`}
                branches={branches}
                onChange={(target_branch) => change({ target_branch })}
              />
            </div>
            <div className="flex gap-6">
              <Toggle label="Review" checked={draft.review} onChange={(review) => change({ review })} />
              <Toggle label="Worktree" checked={draft.worktree} onChange={(worktree) => change({ worktree })} />
              <Toggle label="Preview" checked={draft.preview} onChange={(preview) => change({ preview })} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              {AGENTS.map((agent) => {
                const selected = draft.models[agent] ?? ""
                const options = modelOptions.some((option) => option.value === selected)
                  ? modelOptions
                  : [...modelOptions, { value: selected, label: `${selected} (saved, not added)` }]
                return (
                  <Field key={agent} label={`Model of ${agent}`}>
                    <StyledSelect ariaLabel={`Model of ${agent}`} value={selected} onChange={(value) => changeModel(agent, value)} options={options} />
                  </Field>
                )
              })}
            </div>
            <ErrorText>{branchError}</ErrorText>
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
