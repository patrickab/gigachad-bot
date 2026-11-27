"use client"

import { useCallback, useRef, useState } from "react"
import { Factory } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { cn, errorMessage } from "@/lib/utils"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { Field } from "@/components/ui/Field"
import { AttachImageButton, ImageChips, useRequestImages } from "./requestImages"
import { launchTask } from "./useFactoryTasks"
import { AGENTS, agentModelOptions, BranchSelect, branchOrDefault, loadBranchNames, loadFactoryModels, loadRepoSettings, MERGE_OPTIONS, NO_FACTORY_MODELS, PillChoice, SwitchChoice } from "./FactorySettingsDialog"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import type { FactoryMergeMode, FactoryModels, FactoryResolvedSettings, FactoryTaskOverrides } from "@/lib/domain/factory"

const INHERIT = ""

/** What the user picked in the dialog, null (or a missing model) where the repo's setting applies. */
interface TaskChoices {
  branch: string | null
  merge: FactoryMergeMode | null
  review: boolean | null
  models: FactoryModels
}

/** The overrides of a new task: only the choices that differ from the repo's `settings`. */
export function changedOverrides(settings: FactoryResolvedSettings, choices: TaskChoices): FactoryTaskOverrides {
  const models = Object.fromEntries(Object.entries(choices.models).filter(([agent, model]) => model !== settings.models[agent as keyof FactoryModels]))
  return {
    ...(choices.branch !== null && choices.branch !== settings.branch ? { branch: choices.branch } : {}),
    ...(choices.merge !== null && choices.merge !== settings.merge ? { merge: choices.merge } : {}),
    ...(choices.review !== null && choices.review !== settings.review ? { review: choices.review } : {}),
    ...(Object.keys(models).length > 0 ? { models } : {}),
  }
}

/**
 * Modal that starts a factory task in the repo at path `repo` from a title and a request, in a worktree off a branch
 * it also merges into.
 *
 * - Images (picked or pasted into the request) are stored on the host when Start is pressed, their paths join the request
 * - Start closes the dialog at once: the start runs on as a launch on the board (`launchTask`), which shows a refusal
 * - Everything starts at the repo's settings. Only a choice that differs from them is sent as an override ("Repo
 *   default" names the setting)
 * - The model selects offer the factory models list; while it is empty the core refuses a start, so Start is off
 */
export function NewTaskDialog({ repo, open, onClose }: {
  repo: string
  open: boolean
  onClose: () => void
}) {
  /** The repo's settings over the defaults: what the task runs with where the dialog sends nothing. */
  const [settings, setSettings] = useState<FactoryResolvedSettings | null>(null)
  const [title, setTitle] = useState("")
  const [request, setRequest] = useState("")
  /** The branch the user picked, empty for the repo's setting. */
  const [branch, setBranch] = useState("")
  const [branches, setBranches] = useState<string[] | null>(null)
  const [mergeChoice, setMergeChoice] = useState<FactoryMergeMode | null>(null)
  const [models, setModels] = useState<FactoryModels>({})
  /** The factory models list, null until loaded. */
  const [factoryModels, setFactoryModels] = useState<string[] | null>(null)
  const [error, setError] = useState("")
  const pictures = useRequestImages()
  const titleRef = useRef<HTMLInputElement>(null)
  /** The user's choices, null while the repo's setting applies. */
  const [reviewChoice, setReviewChoice] = useState<boolean | null>(null)
  const review = reviewChoice ?? settings?.review ?? false
  const merge = mergeChoice ?? settings?.merge ?? "review"

  // Every opening starts from an empty form and the repo's saved settings.
  useResetOnOpen(open, () => {
    setSettings(null)
    setTitle("")
    setRequest("")
    setReviewChoice(null)
    setBranch("")
    setBranches(null)
    setMergeChoice(null)
    setModels({})
    setFactoryModels(null)
    setError("")
    pictures.clear()
  })

  const loadSettings = useCallback(async () => (await loadRepoSettings(repo)).resolved, [repo])
  useLoadWhile(open, loadSettings, setSettings, (cause) => setError(errorMessage(cause, "Could not load the settings")))
  const loadBranches = useCallback(() => loadBranchNames(repo), [repo])
  useLoadWhile(open, loadBranches, setBranches, (cause) => setError(errorMessage(cause, "Could not load the branches")))
  useLoadWhile(open, loadFactoryModels, setFactoryModels, (cause) => setError(errorMessage(cause, "Could not load the factory models")))

  const noFactoryModels = factoryModels?.length === 0
  const { run: start, pending: starting, error: startError } = useAsyncAction(async (overrides: FactoryTaskOverrides) => {
    // Without images nothing waits, so the dialog closes at once
    const text = pictures.images.length > 0 ? await pictures.attach(request.trim()) : request.trim()
    void launchTask({ repo, title: title.trim(), request: text, ...(Object.keys(overrides).length > 0 ? { overrides } : {}) })
    onClose()
  }, "Could not store the images")
  const canSubmit = !starting && settings !== null && !noFactoryModels && title.trim() !== "" && request.trim() !== ""
  const submit = () => {
    if (!canSubmit || settings === null) return
    const overrides = changedOverrides(settings, { branch: branchOrDefault(branch), merge: mergeChoice, review: reviewChoice, models })
    void start(overrides)
  }

  return (
    <Dialog open={open} onClose={onClose} icon={Factory} title="New factory task" initialFocusRef={titleRef} widthClassName="max-w-lg">
      <form
        onSubmit={(event) => { event.preventDefault(); submit() }}
        className="flex flex-col gap-3 px-5 pb-5"
      >
        <input
          ref={titleRef}
          type="text"
          aria-label="Title"
          placeholder="Title"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          className={FIELD_CLASS}
          autoComplete="off"
          spellCheck={false}
        />
        <textarea
          aria-label="Request"
          placeholder="Request"
          value={request}
          onChange={(event) => setRequest(event.target.value)}
          onPaste={pictures.onPaste}
          rows={6}
          className={cn(FIELD_CLASS, "resize-none")}
        />
        <ImageChips images={pictures.images} onRemove={pictures.remove} />
        <div className="grid grid-cols-2 gap-3">
          {AGENTS.map((agent) => (
            <Field key={agent} label={agent === "big" ? "Big model" : "Reviewer"}>
              <StyledSelect
                quiet
                ariaLabel={`Model of ${agent}`}
                value={models[agent] ?? INHERIT}
                disabled={settings === null || (agent === "reviewer" && !review)}
                onChange={(value) => setModels((current) => {
                  const next = { ...current }
                  if (value === INHERIT) delete next[agent]
                  else next[agent] = value
                  return next
                })}
                options={agentModelOptions(`Repo default (${settings?.models[agent] ?? "Pi default"})`, factoryModels ?? [], models[agent] ?? INHERIT)}
              />
            </Field>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-2">
          <PillChoice label="Merge" options={MERGE_OPTIONS} value={merge} disabled={settings === null} onChange={setMergeChoice} />
          <SwitchChoice label="Review" checked={review} disabled={settings === null} onChange={setReviewChoice} />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <BranchSelect
            label="Branch"
            value={branch}
            fallback={`Repo default${settings ? ` (${settings.branch})` : ""}`}
            branches={branches}
            disabled={settings === null}
            onChange={setBranch}
          />
        </div>
        <ErrorText>{error || startError || (noFactoryModels ? NO_FACTORY_MODELS : "")}</ErrorText>
        <div className="flex items-center justify-end gap-2 pt-1">
          <AttachImageButton onPick={pictures.add} disabled={starting} className={cn(QUIET_BUTTON_CLASS, "mr-auto")} />
          <button type="button" onClick={onClose} className={QUIET_BUTTON_CLASS}>Cancel</button>
          <button type="submit" disabled={!canSubmit} className={PRIMARY_BUTTON_CLASS}>Start</button>
        </div>
      </form>
    </Dialog>
  )
}
