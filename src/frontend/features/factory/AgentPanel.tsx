"use client"

import { useState } from "react"
import { Bot, ListChecks } from "lucide-react"
import { PillTabs } from "@/components/ui/PillTabs"
import { FilePanel } from "@/features/editor/FilePanel"
import type { PiShellCommands } from "@/features/pi/PiTab"
import type { PiCatalogProps } from "@/features/pi/PiTopBar"
import type { FactoryEntry, FactoryTask } from "@/lib/domain/factory"
import { TaskPiChat } from "./TaskPiChat"
import { TaskTimeline } from "./TaskTimeline"

const PANEL_WIDTH_PX = 420

type PanelMode = "timeline" | "session"

const MODES: readonly { value: PanelMode; label: React.ReactNode; ariaLabel: string }[] = [
  { value: "timeline", label: <ListChecks aria-hidden className="h-3.5 w-3.5" />, ariaLabel: "Timeline" },
  { value: "session", label: <Bot aria-hidden className="h-3.5 w-3.5" />, ariaLabel: "Agent session" },
]

/**
 * A task's agent in the tab's file panel, beside the diff.
 *
 * - Timeline mode: the pipeline's rounds, from `entries` (the owner reads them, the diff follows them too)
 * - Session mode: the task's own Pi session (`TaskPiChat`), to read it and steer the coder. Its bar fills the header
 */
export function AgentPanel({ task, entries, timelineError, onClose, modelProps, shellCommands }: {
  task: FactoryTask
  entries: readonly FactoryEntry[]
  timelineError: string
  onClose: () => void
  modelProps: PiCatalogProps
  shellCommands?: PiShellCommands
}) {
  const [mode, setMode] = useState<PanelMode>("timeline")
  const [bar, setBar] = useState<HTMLDivElement | null>(null)

  const actions = <>
    <PillTabs className="shrink-0" pillClassName="px-2" options={MODES} value={mode} onChange={setMode} tabs={{ label: "Agent view", idPrefix: "agent-mode", panelId: "agent-mode-panel" }} />
    <div ref={setBar} className="flex min-w-0 flex-1" />
  </>

  return (
    <FilePanel label="Agent" bare onClose={onClose} width={PANEL_WIDTH_PX} actions={actions}>
      <div id="agent-mode-panel" role="tabpanel" aria-labelledby={`agent-mode-${mode}`} className="flex min-h-0 flex-1 flex-col">
        {mode === "session"
          ? <TaskPiChat task={task} bar={bar} modelProps={modelProps} shellCommands={shellCommands} />
          : <TaskTimeline entries={entries} error={timelineError} />}
      </div>
    </FilePanel>
  )
}
