"use client"

import { useCallback, useRef, useState } from "react"
import { Factory } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { cn, errorMessage } from "@/lib/utils"
import { fetchModels, getFactorySettings, listFactoryBranches } from "@/lib/backend/api"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { Field } from "@/components/ui/Field"
import { launchTask } from "./useFactoryTasks"
import { AGENTS, BranchSelect, branchOrDefault, factoryModelIds, MERGE_OPTIONS, PillChoice, SOURCE_BRANCH_DEFAULT, SwitchChoice, TARGET_BRANCH_DEFAULT, WORKSPACE_OPTIONS } from "./FactorySettingsDialog"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import type { FactoryMergeMode, FactoryModels, FactorySettings } from "@/lib/domain/factory"
import type { ModelsResponse } from "@/lib/domain/types"

const INHERIT = ""

/** The branch overrides of a worktree task: each branch the user picked, none for the repo default. */
function branchOverrides(source: string, target: string): Partial<FactorySettings> {
  const source_branch = branchOrDefault(source)
  const target_branch = branchOrDefault(target)
  return { ...(source_branch ? { source_branch } : {}), ...(target_branch ? { target_branch } : {}) }
}


/**
 * Modal that starts a factory task in the repo at path `repo` from a title and a request, in a worktree off a source
 * branch that merges into a target branch, or in the current checkout.
 *
 * - Start closes the dialog at once: the start runs on as a launch on the board (`launchTask`), which shows a refusal
 * - Everything starts at the repo's settings. The branches and models are sent only when picked ("Repo default" names
 *   the setting), the merge mode only when changed, the branches only for a worktree task (a current-checkout task works
 *   on what is checked out)
 */
export function NewTaskDialog({ repo, open, onClose }: {
  repo: string
  open: boolean
  onClose: () => void
}) {
  const [settings, setSettings] = useState<FactorySettings | null>(null)
  const [title, setTitle] = useState("")
  const [request, setRequest] = useState("")
  /** The user's choice, null while the repo's setting applies. */
  const [worktreeChoice, setWorktreeChoice] = useState<boolean | null>(null)
  /** The branches the user picked, empty for the repo's setting. */
  const [sourceBranch, setSourceBranch] = useState("")
  const [targetBranch, setTargetBranch] = useState("")
  const [branches, setBranches] = useState<string[] | null>(null)
  const [mergeChoice, setMergeChoice] = useState<FactoryMergeMode | null>(null)
  const [models, setModels] = useState<FactoryModels>({})
  const [catalog, setCatalog] = useState<ModelsResponse | null>(null)
  const [error, setError] = useState("")
  const titleRef = useRef<HTMLInputElement>(null)
  const worktree = worktreeChoice ?? settings?.worktree ?? false
  const [reviewChoice, setReviewChoice] = useState<boolean | null>(null)
  const review = reviewChoice ?? settings?.review ?? false
  const merge = mergeChoice ?? settings?.merge ?? "review"

  // Every opening starts from an empty form and the repo's saved settings.
  useResetOnOpen(open, () => {
    setSettings(null)
    setTitle("")
    setRequest("")
    setWorktreeChoice(null)
    setReviewChoice(null)
    setSourceBranch("")
    setTargetBranch("")
    setBranches(null)
    setMergeChoice(null)
    setModels({})
    setError("")
  })

  const loadSettings = useCallback(() => getFactorySettings(repo), [repo])
  useLoadWhile(open, loadSettings, setSettings, (cause) => setError(errorMessage(cause, "Could not load the settings")))
  const loadBranches = useCallback(() => listFactoryBranches(repo), [repo])
  useLoadWhile(open, loadBranches, setBranches, (cause) => setError(errorMessage(cause, "Could not load the branches")))
  useLoadWhile(open, fetchModels, setCatalog, (cause) => setError(errorMessage(cause, "Could not load Pi models")))

  const canSubmit = settings !== null && title.trim() !== "" && request.trim() !== ""
  const submit = () => {
    if (!canSubmit) return
    void launchTask({ repo, title: title.trim(), request: request.trim(), overrides: {
      worktree,
      review,
      ...(worktree ? branchOverrides(sourceBranch, targetBranch) : {}),
      ...(mergeChoice !== null ? { merge: mergeChoice } : {}),
      ...(Object.keys(models).length > 0 ? { models } : {}),
    } })
    onClose()
  }

  const modelIds = factoryModelIds(catalog)

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
          rows={6}
          className={cn(FIELD_CLASS, "resize-none")}
        />
        <div className="grid grid-cols-2 gap-3">
          {AGENTS.map((agent) => (
            <Field key={agent} label={agent === "coder" ? "Coder" : "Reviewer"}>
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
                options={[
                  { value: INHERIT, label: `Repo default (${settings?.models[agent] ?? "Pi default"})` },
                  ...modelIds.map((model) => ({ value: model, label: model })),
                ]}
              />
            </Field>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-2">
          <PillChoice label="Workspace" options={WORKSPACE_OPTIONS} value={worktree ? "worktree" : "checkout"} disabled={settings === null} onChange={(workspace) => setWorktreeChoice(workspace === "worktree")} />
          <SwitchChoice label="Review" checked={review} disabled={settings === null} onChange={setReviewChoice} />
          <PillChoice label="Merge" options={MERGE_OPTIONS} value={merge} disabled={settings === null} onChange={setMergeChoice} />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <BranchSelect
            label="Source branch"
            value={sourceBranch}
            fallback={`Repo default (${settings?.source_branch ?? SOURCE_BRANCH_DEFAULT})`}
            branches={branches}
            disabled={settings === null || !worktree}
            onChange={setSourceBranch}
          />
          <BranchSelect
            label="Target branch"
            value={targetBranch}
            fallback={`Repo default (${settings?.target_branch ?? TARGET_BRANCH_DEFAULT})`}
            branches={branches}
            disabled={settings === null || !worktree}
            onChange={setTargetBranch}
          />
        </div>
        <ErrorText>{error}</ErrorText>
        <div className="flex items-center justify-end gap-2 pt-1">
          <button type="button" onClick={onClose} className={QUIET_BUTTON_CLASS}>Cancel</button>
          <button type="submit" disabled={!canSubmit} className={PRIMARY_BUTTON_CLASS}>Start</button>
        </div>
      </form>
    </Dialog>
  )
}
