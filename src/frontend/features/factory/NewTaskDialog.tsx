"use client"

import { useCallback, useRef, useState } from "react"
import { ChevronRight, Factory } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { cn, errorMessage } from "@/lib/utils"
import { createFactoryTask, fetchFactoryWorkflows } from "@/lib/backend/api"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { Field } from "@/components/ui/Field"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import type { FactoryOverrides, FactoryWorkflow, MergeMode } from "@/lib/domain/factory"

const INHERIT = ""

type SandboxChoice = typeof INHERIT | "on" | "off"

const SANDBOX_OPTIONS = [
  { value: INHERIT, label: "Repo default" },
  { value: "on", label: "On" },
  { value: "off", label: "Off" },
]

/** Overrides the user set; untouched fields are omitted so the workflow and repo defaults apply. */
function buildOverrides(merge: MergeMode | typeof INHERIT, sandbox: SandboxChoice): FactoryOverrides {
  return {
    ...(merge !== INHERIT ? { merge } : {}),
    ...(sandbox !== INHERIT ? { sandbox: sandbox === "on" } : {}),
  }
}

/** Modal that creates a factory task from a workflow, a title and a request, started at once or saved to the backlog. */
export function NewTaskDialog({ repo, open, onClose }: {
  repo: string
  open: boolean
  onClose: () => void
}) {
  const [workflows, setWorkflows] = useState<FactoryWorkflow[] | null>(null)
  const [workflowId, setWorkflowId] = useState("")
  const [title, setTitle] = useState("")
  const [request, setRequest] = useState("")
  const [merge, setMerge] = useState<MergeMode | typeof INHERIT>(INHERIT)
  const [sandbox, setSandbox] = useState<SandboxChoice>(INHERIT)
  const [overridesOpen, setOverridesOpen] = useState(false)
  const titleRef = useRef<HTMLInputElement>(null)
  const { run: create, pending: busy, error, setError } = useAsyncAction(async (start: boolean) => {
    await createFactoryTask({ repo, workflow: workflowId, title: title.trim(), request: request.trim(), overrides: buildOverrides(merge, sandbox), start })
    onClose()
  }, "Could not create the task")

  // Every opening starts from an empty form.
  useResetOnOpen(open, () => {
    setTitle("")
    setRequest("")
    setMerge(INHERIT)
    setSandbox(INHERIT)
    setOverridesOpen(false)
    setError("")
  })

  const loadWorkflows = useCallback(() => fetchFactoryWorkflows(repo), [repo])
  useLoadWhile(open, loadWorkflows, (loaded) => {
    setWorkflows(loaded)
    setWorkflowId((current) => loaded.some((workflow) => workflow.id === current) ? current : loaded[0]?.id ?? "")
  }, (cause) => setError(errorMessage(cause, "Could not load workflows")))

  const workflow = workflows?.find((candidate) => candidate.id === workflowId)
  const canSubmit = !busy && workflow !== undefined && title.trim() !== "" && request.trim() !== ""
  const submit = (start: boolean) => {
    if (canSubmit) void create(start)
  }

  const mergeOptions = [
    { value: INHERIT, label: workflow ? `Workflow default (${workflow.merge})` : "Workflow default" },
    { value: "review", label: "Review" },
    { value: "auto", label: "Auto" },
  ]

  return (
    <Dialog open={open} onClose={onClose} icon={Factory} title="New factory task" initialFocusRef={titleRef} widthClassName="max-w-lg">
      <form
        onSubmit={(event) => { event.preventDefault(); submit(true) }}
        className="flex flex-col gap-3 px-5 pb-5"
      >
        <Field label="Workflow">
          <StyledSelect
            ariaLabel="Workflow"
            value={workflowId}
            onChange={setWorkflowId}
            options={(workflows ?? []).map((candidate) => ({ value: candidate.id, label: candidate.name }))}
          />
          {workflow && <span className="text-[11px] text-ink-faint">{workflow.phases.join(" → ")}</span>}
        </Field>
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
          placeholder="What should the agent do?"
          value={request}
          onChange={(event) => setRequest(event.target.value)}
          rows={6}
          className={cn(FIELD_CLASS, "resize-none")}
        />
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
                <StyledSelect ariaLabel="Merge" value={merge} onChange={(value) => setMerge(value as MergeMode | typeof INHERIT)} options={mergeOptions} />
              </Field>
              <Field label="Sandbox">
                <StyledSelect ariaLabel="Sandbox" value={sandbox} onChange={(value) => setSandbox(value as SandboxChoice)} options={SANDBOX_OPTIONS} />
              </Field>
            </div>
          )}
        </div>
        <ErrorText>{error}</ErrorText>
        <div className="flex items-center justify-end gap-2 pt-1">
          <button type="button" onClick={onClose} className={QUIET_BUTTON_CLASS}>Cancel</button>
          <button type="button" disabled={!canSubmit} onClick={() => submit(false)} className={QUIET_BUTTON_CLASS}>Save to backlog</button>
          <button type="submit" disabled={!canSubmit} className={PRIMARY_BUTTON_CLASS}>Start</button>
        </div>
      </form>
    </Dialog>
  )
}
