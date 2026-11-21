"use client"

import { useCallback, useRef, useState } from "react"
import { ChevronRight, Factory } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { cn, errorMessage } from "@/lib/utils"
import { createFactoryTask, getFactorySettings } from "@/lib/backend/api"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { Field } from "@/components/ui/Field"
import { branchOrDefault, Toggle } from "./FactorySettingsDialog"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import type { FactoryMergeMode, FactorySettings } from "@/lib/domain/factory"

const INHERIT = ""

type MergeChoice = FactoryMergeMode | typeof INHERIT

/**
 * Modal that starts a factory task in the repo at path `repo` from a title and a request, in a worktree off a branch or
 * in the current checkout.
 *
 * - The branch is sent only for a worktree task (a current-checkout task works on what is checked out), null when empty
 */
export function NewTaskDialog({ repo, open, onClose, onCreated }: {
  repo: string
  open: boolean
  onClose: () => void
  /** Called once the task exists, before the dialog closes. */
  onCreated?: () => void
}) {
  const [settings, setSettings] = useState<FactorySettings | null>(null)
  const [title, setTitle] = useState("")
  const [request, setRequest] = useState("")
  /** The user's choice, null while the repo's setting applies. */
  const [worktreeChoice, setWorktreeChoice] = useState<boolean | null>(null)
  /** The branch field as the user typed it, null while the repo's setting fills it. */
  const [branchChoice, setBranchChoice] = useState<string | null>(null)
  const [merge, setMerge] = useState<MergeChoice>(INHERIT)
  const [overridesOpen, setOverridesOpen] = useState(false)
  const titleRef = useRef<HTMLInputElement>(null)
  const worktree = worktreeChoice ?? settings?.worktree ?? false
  const branch = branchChoice ?? settings?.branch ?? ""
  const { run: create, pending: busy, error, setError } = useAsyncAction(async () => {
    await createFactoryTask({ repo, title: title.trim(), request: request.trim(), overrides: {
      worktree,
      ...(worktree ? { branch: branchOrDefault(branch) } : {}),
      ...(merge !== INHERIT ? { merge } : {}),
    } })
    onCreated?.()
    onClose()
  }, "Could not create the task")

  // Every opening starts from an empty form and the repo's saved settings.
  useResetOnOpen(open, () => {
    setSettings(null)
    setTitle("")
    setRequest("")
    setWorktreeChoice(null)
    setBranchChoice(null)
    setMerge(INHERIT)
    setOverridesOpen(false)
    setError("")
  })

  const loadSettings = useCallback(() => getFactorySettings(repo), [repo])
  useLoadWhile(open, loadSettings, setSettings, (cause) => setError(errorMessage(cause, "Could not load the settings")))

  const canSubmit = !busy && settings !== null && title.trim() !== "" && request.trim() !== ""
  const submit = () => {
    if (canSubmit) void create()
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
        <div className="grid grid-cols-2 items-end gap-3">
          <div className="py-2"><Toggle label="Worktree" checked={worktree} disabled={settings === null} onChange={setWorktreeChoice} /></div>
          <Field label="Branch">
            <input
              type="text"
              aria-label="Branch"
              value={branch}
              disabled={settings === null || !worktree}
              onChange={(event) => setBranchChoice(event.target.value)}
              className={cn(FIELD_CLASS, "disabled:opacity-40")}
              autoComplete="off"
              spellCheck={false}
            />
          </Field>
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
