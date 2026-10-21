"use client"

import { useEffect, useRef, useState } from "react"
import { ChevronRight, Factory } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { cn } from "@/lib/utils"
import { createFactoryTask, fetchFactoryWorkflows } from "@/lib/backend/api"
import { FIELD_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import type { FactoryOverrides, FactoryWorkflow, MergeMode } from "@/lib/domain/types"

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
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const titleRef = useRef<HTMLInputElement>(null)
  const [wasOpen, setWasOpen] = useState(open)

  // Every opening starts from an empty form.
  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) {
      setTitle("")
      setRequest("")
      setMerge(INHERIT)
      setSandbox(INHERIT)
      setOverridesOpen(false)
      setError("")
    }
  }

  useEffect(() => {
    if (!open) return
    let cancelled = false
    fetchFactoryWorkflows(repo).then((loaded) => {
      if (cancelled) return
      setWorkflows(loaded)
      setWorkflowId((current) => loaded.some((workflow) => workflow.id === current) ? current : loaded[0]?.id ?? "")
    }).catch((cause: unknown) => {
      if (!cancelled) setError(cause instanceof Error ? cause.message : "Could not load workflows")
    })
    return () => { cancelled = true }
  }, [open, repo])

  const workflow = workflows?.find((candidate) => candidate.id === workflowId)
  const canSubmit = !busy && workflow !== undefined && title.trim() !== "" && request.trim() !== ""

  const submit = async (start: boolean) => {
    if (!canSubmit) return
    setBusy(true)
    setError("")
    try {
      await createFactoryTask({ repo, workflow: workflowId, title: title.trim(), request: request.trim(), overrides: buildOverrides(merge, sandbox), start })
      onClose()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not create the task")
    } finally {
      setBusy(false)
    }
  }

  const mergeOptions = [
    { value: INHERIT, label: workflow ? `Workflow default (${workflow.merge})` : "Workflow default" },
    { value: "review", label: "Review" },
    { value: "auto", label: "Auto" },
  ]

  return (
    <Dialog open={open} onClose={onClose} icon={Factory} title="New factory task" initialFocusRef={titleRef} widthClassName="max-w-lg">
      <form
        onSubmit={(event) => { event.preventDefault(); void submit(true) }}
        className="flex flex-col gap-3 px-5 pb-5"
      >
        <div className="flex flex-col gap-1">
          <span className="text-[11px] text-ink-subtle">Workflow</span>
          <StyledSelect
            ariaLabel="Workflow"
            value={workflowId}
            onChange={setWorkflowId}
            options={(workflows ?? []).map((candidate) => ({ value: candidate.id, label: candidate.name }))}
          />
          {workflow && <span className="text-[11px] text-ink-faint">{workflow.phases.join(" → ")}</span>}
        </div>
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
              <div className="flex flex-col gap-1">
                <span className="text-[11px] text-ink-subtle">Merge</span>
                <StyledSelect ariaLabel="Merge" value={merge} onChange={(value) => setMerge(value as MergeMode | typeof INHERIT)} options={mergeOptions} />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-[11px] text-ink-subtle">Sandbox</span>
                <StyledSelect ariaLabel="Sandbox" value={sandbox} onChange={(value) => setSandbox(value as SandboxChoice)} options={SANDBOX_OPTIONS} />
              </div>
            </div>
          )}
        </div>
        {error && <p role="alert" className="text-xs text-danger">{error}</p>}
        <div className="flex items-center justify-end gap-2 pt-1">
          <button type="button" onClick={onClose} className={QUIET_BUTTON_CLASS}>Cancel</button>
          <button type="button" disabled={!canSubmit} onClick={() => void submit(false)} className={QUIET_BUTTON_CLASS}>Save to backlog</button>
          <button type="submit" disabled={!canSubmit} className={PRIMARY_BUTTON_CLASS}>Start</button>
        </div>
      </form>
    </Dialog>
  )
}
