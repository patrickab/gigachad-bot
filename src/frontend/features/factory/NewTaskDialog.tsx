"use client"

import { useCallback, useRef, useState } from "react"
import { ChevronRight, Factory } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { cn, errorMessage } from "@/lib/utils"
import { getFactorySettings } from "@/lib/backend/api"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { Field } from "@/components/ui/Field"
import { launchTask } from "./useFactoryTasks"
import { BranchField, branchOrDefault, SOURCE_BRANCH_PLACEHOLDER, TARGET_BRANCH_PLACEHOLDER, Toggle } from "./FactorySettingsDialog"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import type { FactoryMergeMode, FactorySettings } from "@/lib/domain/factory"

const INHERIT = ""

/** The branch overrides of a worktree task: each branch the user filled in, none for an empty field. */
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
 *   when filled in: an empty field keeps the repo's setting, which its placeholder names
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
  /** The branch fields as the user typed them, empty for the repo's setting. */
  const [sourceBranch, setSourceBranch] = useState("")
  const [targetBranch, setTargetBranch] = useState("")
  const [merge, setMerge] = useState<MergeChoice>(INHERIT)
  const [overridesOpen, setOverridesOpen] = useState(false)
  const [error, setError] = useState("")
  const titleRef = useRef<HTMLInputElement>(null)
  const worktree = worktreeChoice ?? settings?.worktree ?? false

  // Every opening starts from an empty form and the repo's saved settings.
  useResetOnOpen(open, () => {
    setSettings(null)
    setTitle("")
    setRequest("")
    setWorktreeChoice(null)
    setSourceBranch("")
    setTargetBranch("")
    setMerge(INHERIT)
    setOverridesOpen(false)
    setError("")
  })

  const loadSettings = useCallback(() => getFactorySettings(repo), [repo])
  useLoadWhile(open, loadSettings, setSettings, (cause) => setError(errorMessage(cause, "Could not load the settings")))

  const canSubmit = settings !== null && title.trim() !== "" && request.trim() !== ""
  const submit = () => {
    if (!canSubmit) return
    void launchTask({ repo, title: title.trim(), request: request.trim(), overrides: {
      worktree,
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
        <Toggle label="Worktree" checked={worktree} disabled={settings === null} onChange={setWorktreeChoice} />
        <div className="grid grid-cols-2 gap-3">
          <BranchField
            label="Source branch"
            value={sourceBranch}
            placeholder={settings?.source_branch ?? SOURCE_BRANCH_PLACEHOLDER}
            disabled={settings === null || !worktree}
            onChange={setSourceBranch}
          />
          <BranchField
            label="Target branch"
            value={targetBranch}
            placeholder={settings?.target_branch ?? TARGET_BRANCH_PLACEHOLDER}
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
