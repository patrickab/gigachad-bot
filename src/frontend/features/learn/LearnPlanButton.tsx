"use client"

import { ListChecks } from "lucide-react"
import { useTab } from "@/contexts/TabContext"
import type { LearnView } from "@/lib/domain/types"
import { useLearnPanels } from "./useLearnPanels"

/** Top bar button opening LearnHarness's Plan popup for the session `view` shows, like its hidden title bar's PLAN chip. Only a session with a plan has one. */
export function LearnPlanButton({ view }: { view: LearnView }) {
  const { tab } = useTab()
  const { hasPlan, openPanel } = useLearnPanels()
  if (view.kind !== "session" || !hasPlan(view.sessionId)) return null
  return (
    <button
      type="button"
      aria-label="Plan"
      onClick={() => openPanel({ id: view.sessionId, project_slug: tab.projectSlug }, "plan")}
      className="rounded p-1 text-ink-subtle transition-colors hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
    >
      <ListChecks className="h-4 w-4" />
    </button>
  )
}
