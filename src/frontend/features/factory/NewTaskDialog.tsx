"use client"

import { useCallback, useRef, useState } from "react"
import { ChevronRight, Factory } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { cn, errorMessage } from "@/lib/utils"
import { getFactorySettings, listFactoryBranches } from "@/lib/backend/api"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { Field } from "@/components/ui/Field"
import { launchTask } from "./useFactoryTasks"
import { BranchSelect, branchOrDefault, PillChoice, REVIEW_OPTIONS, SOURCE_BRANCH_DEFAULT, TARGET_BRANCH_DEFAULT, WORKSPACE_OPTIONS } from "./FactorySettingsDialog"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import type { FactoryMergeMode, FactorySettings } from "@/lib/domain/factory"

const INHERIT = ""

/** The branch overrides of a worktree task: each branch the user picked, none for the repo default. */
function branchOverrides(source: string, target: string): Partial<FactorySettings> {
  const source_branch = branchOrDefault(source)
  const target_branch = branchOrDefault(target)
  return { ...(source_branch ? { source_branch } : {}), ...(target_branch ? { target_branch } : {}) }
}

type MergeChoice = FactoryMergeMode | typeof INHERIT

/**
 * Modal that starts a factory task in the repo at path `repo` from a title and a request, in a worktree off a source
 * branch that merges into a target branch, or in the current checkout.
 *
 * - Start closes the dialog at once: the start runs on as a launch on the board (`launchTask`), which shows a refusal
 * - The branches are sent only for a worktree task (a current-checkout task works on what is checked out), and only
 *   when picked: "Repo default", which names the repo's setting, keeps it
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
  const [merge, setMerge] = useState<MergeChoice>(INHERIT)
  const [overridesOpen, setOverridesOpen] = useState(false)
  const [error, setError] = useState("")
  const titleRef = useRef<HTMLInputElement>(null)
  const worktree = worktreeChoice ?? settings?.worktree ?? false
  const [reviewChoice, setReviewChoice] = useState<boolean | null>(null)
  const review = reviewChoice ?? settings?.review ?? false

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
    setMerge(INHERIT)
    setOverridesOpen(false)
    setError("")
  })

  const loadSettings = useCallback(() => getFactorySettings(repo), [repo])
  useLoadWhile(open, loadSettings, setSettings, (cause) => setError(errorMessage(cause, "Could not load the settings")))
  const loadBranches = useCallback(() => listFactoryBranches(repo), [repo])
  useLoadWhile(open, loadBranches, setBranches, (cause) => setError(errorMessage(cause, "Could not load the branches")))

  const canSubmit = settings !== null && title.trim() !== "" && request.trim() !== ""
  const submit = () => {
    if (!canSubmit) return
    void launchTask({ repo, title: title.trim(), request: request.trim(), overrides: {
      worktree,
      review,
      ...(worktree ? branchOverrides(sourceBranch, targetBranch) : {}),
      ...(merge !== INHERIT ? { merge } : {}),
    } })
    onClose()
  }

  const mergeOptions = [
    { value: INHERIT, label: settings ? `Repo default (${settings.merge})` : "Repo default" },
    { value: "review", label: "Review" },
    { value: "auto", label: "Auto" },
  ]

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
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          <PillChoice label="Workspace" options={WORKSPACE_OPTIONS} value={worktree ? "worktree" : "checkout"} disabled={settings === null} onChange={(workspace) => setWorktreeChoice(workspace === "worktree")} />
          <PillChoice label="Review" options={REVIEW_OPTIONS} value={review ? "on" : "off"} disabled={settings === null} onChange={(choice) => setReviewChoice(choice === "on")} />
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
        <div>
          <button
            type="button"
            aria-expanded={overridesOpen}
            onClick={() => setOverridesOpen((current) => !current)}
            className="flex items-center gap-1 rounded text-[11px] text-ink-subtle transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
          >
            <ChevronRight className={cn("h-3 w-3 transition-transform", overridesOpen && "rotate-90")} />
            Overrides
          </button>
          {overridesOpen && (
            <div className="mt-2 grid grid-cols-2 gap-3">
              <Field label="Merge">
                <StyledSelect ariaLabel="Merge" value={merge} onChange={(value) => setMerge(value as MergeChoice)} options={mergeOptions} />
              </Field>
            </div>
          )}
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
