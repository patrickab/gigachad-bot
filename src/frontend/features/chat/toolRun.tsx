"use client"

import { useEffect, useState, type ReactNode } from "react"
import { Loader2 } from "lucide-react"
import type { ToolCall, ToolRun } from "@/features/conversation/tools/types"
import type { ToolCallRecord } from "@/lib/domain/types"

/** The chat's tool call as Pi's row reads it: the call, its run (the full record in `run.result.details`) and its stage timeline. */
export interface ChatToolRun {
  call: ToolCall
  run: ToolRun
  stages: ReactNode
}

// Streaming replaces a record whose call changed and keeps the others (`lib/backend/streaming.ts`), so the record's
// identity covers its status: a streamed frame builds no new run for a settled call.
const adapted = new WeakMap<ToolCallRecord, ChatToolRun>()
const records = new WeakSet<object>()

/** `record` as a Pi tool call and run, built once per record. */
export function chatToolRun(record: ToolCallRecord): ChatToolRun {
  let built = adapted.get(record)
  if (!built) {
    const text = record.status === "error" ? record.error || record.summary || "" : record.status === "done" ? record.summary ?? "" : ""
    records.add(record)
    built = {
      call: { type: "toolCall", id: record.id, name: record.name, arguments: record.arguments },
      run: {
        running: record.status === "running",
        isError: record.status === "error",
        result: { content: text ? [{ type: "text", text }] : [], details: record },
      },
      stages: record.stages?.length ? <ChatToolStages stages={record.stages} /> : undefined,
    }
    adapted.set(record, built)
  }
  return built
}

/** The chat record behind `run`, undefined for a run of a Pi session. */
export function chatRecordOf(run: ToolRun | undefined): ToolCallRecord | undefined {
  const details = run?.result?.details
  return typeof details === "object" && details !== null && records.has(details) ? details as ToolCallRecord : undefined
}

type Stage = NonNullable<ToolCallRecord["stages"]>[number]

/** Elapsed time of a stage. A running stage ticks locally from `started_at` (epoch seconds), never below the server-reported duration. */
function stageDuration(stage: Stage, now: number): string {
  const seconds = stage.status === "running" ? Math.max(stage.duration, now / 1_000 - stage.started_at) : stage.duration
  return `${seconds.toFixed(1)}s`
}

/** A call's stages with their durations; a running stage's timer ticks. */
function ChatToolStages({ stages }: { stages: Stage[] }) {
  const [now, setNow] = useState(() => Date.now())
  const running = stages.some((stage) => stage.status === "running")
  // Re-render at 10 Hz only while a stage runs so its timer ticks.
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => setNow(Date.now()), 100)
    return () => window.clearInterval(timer)
  }, [running])
  return stages.map((stage) => (
    <span key={stage.id} className="flex shrink-0 items-center gap-1.5">
      {stage.status === "running" && <Loader2 className="h-3 w-3 animate-spin text-ink-faint" aria-hidden="true" />}
      <span className="tabular-nums text-ink-faint">{stageDuration(stage, now)}</span>
      <span className={stage.status === "running" ? "text-ink" : stage.status === "error" ? "text-danger" : "text-ink-muted"}>
        {stage.status === "error" ? "Failed" : stage.label}
      </span>
    </span>
  ))
}
