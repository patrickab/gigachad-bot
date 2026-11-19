"use client"

import { StatusDot, type StatusTone } from "@/components/ui/StatusDot"
import { PiModelSelector, PiThinkingSelector } from "./PiModelSelector"
import type { PiModel } from "./types"
import type { PiSessionStats } from "./usePiSession"

/** The session's model, thinking and usage as its header shows and changes them. */
export interface PiSessionControls {
  model: PiModel | null
  models: PiModel[]
  thinkingLevel: string | null
  /** Levels the model offers, `["off"]` (no selector) for a model without reasoning. */
  thinkingLevels: string[]
  stats: PiSessionStats | null
  onModel: (model: PiModel) => void
  onThinking: (level: string) => void
}

/** What a session view's header shows. */
export interface PiSessionHeader {
  tone: StatusTone
  sessionName: string | undefined
  controls: PiSessionControls
}

function compactTokens(tokens: number): string {
  return tokens >= 1000 ? `${Math.round(tokens / 1000)}k` : String(tokens)
}

/** `30% · $0.45`: context window use (unknown right after a compaction) and the session's cost, null before Pi reports them (F3). */
export function statsText(stats: PiSessionStats | null): { text: string; label: string } | null {
  if (!stats) return null
  const cost = `$${stats.cost.toFixed(2)}`
  const usage = stats.contextUsage
  if (!usage) return { text: cost, label: `Cost ${cost}` }
  const percent = usage.percent === null ? "?" : `${Math.round(usage.percent)}%`
  const tokens = usage.tokens === null ? "unknown" : compactTokens(usage.tokens)
  return { text: `${percent} · ${cost}`, label: `Context ${tokens} of ${compactTokens(usage.contextWindow)} tokens, cost ${cost}` }
}

/** Model and thinking selectors and the usage line, for a session header. */
export function PiSessionControlsBar({ controls }: { controls: PiSessionControls }) {
  const stats = statsText(controls.stats)
  return (
    <div className="flex shrink-0 items-center gap-1">
      {controls.models.length > 0 && <PiModelSelector models={controls.models} selected={controls.model} onSelect={controls.onModel} />}
      {controls.thinkingLevel && controls.thinkingLevels.length > 1 && (
        <PiThinkingSelector levels={controls.thinkingLevels} selected={controls.thinkingLevel} onSelect={controls.onThinking} />
      )}
      {stats && <span aria-label={stats.label} className="px-1.5 font-mono text-xs tabular-nums text-ink-faint">{stats.text}</span>}
    </div>
  )
}

/** A Pi tab's top bar: live dot, repo, session name, then the session's controls once it runs. */
export function PiTopBar({ repo, header }: { repo: string; header: PiSessionHeader | null }) {
  return (
    <div className="flex min-w-0 flex-1 items-center gap-2 px-2 text-sm">
      {header && <StatusDot tone={header.tone} />}
      <span className="font-medium text-ink">{repo}</span>
      {header?.sessionName && <span className="truncate text-ink-muted">{header.sessionName}</span>}
      {header && <div className="ml-auto"><PiSessionControlsBar controls={header.controls} /></div>}
    </div>
  )
}
