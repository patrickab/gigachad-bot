"use client"

import type { ReactNode } from "react"
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

function StatsLine({ stats }: { stats: PiSessionStats | null }) {
  const shown = statsText(stats)
  return shown && <span aria-label={shown.label} className="shrink-0 whitespace-nowrap px-1.5 font-mono text-xs tabular-nums text-ink-faint">{shown.text}</span>
}

/** Model and thinking selectors, then the usage line unless `usage` is false, for a session header. `compact` shortens the model name for a side panel. */
export function PiSessionControlsBar({ controls, compact = false, usage = true }: { controls: PiSessionControls; compact?: boolean; usage?: boolean }) {
  return (
    <div className={compact ? "flex min-w-0 items-center gap-1" : "flex shrink-0 items-center gap-1"}>
      {controls.models.length > 0 && <PiModelSelector models={controls.models} selected={controls.model} onSelect={controls.onModel} compact={compact} />}
      {controls.thinkingLevel && controls.thinkingLevels.length > 1 && (
        <div className="shrink-0">
          <PiThinkingSelector levels={controls.thinkingLevels} selected={controls.thinkingLevel} onSelect={controls.onThinking} />
        </div>
      )}
      {usage && <StatsLine stats={controls.stats} />}
    </div>
  )
}

/** A Pi tab's top bar: session name, then the session's model and thinking once it runs, or `controls` (what a new session starts on) before. */
export function PiTopBar({ header, controls }: { header: PiSessionHeader | null; controls?: PiSessionControls }) {
  const shown = header?.controls ?? controls
  return (
    <div className="flex min-w-0 flex-1 items-center gap-2 px-2 text-sm">
      {header?.sessionName && <span className="truncate text-ink-muted">{header.sessionName}</span>}
      {shown && <PiSessionControlsBar controls={shown} usage={false} />}
    </div>
  )
}

/** The top bar's right end, as omp-web's: the context's `Tokens` over their count, then context use and cost. Nothing before Pi reports them (F3). */
export function PiUsageBar({ stats }: { stats: PiSessionStats | null }) {
  if (!stats) return null
  const tokens = stats.contextUsage?.tokens
  return (
    <div className="flex items-center gap-1 px-2">
      {stats.contextUsage && (
        <div className="flex flex-col items-center px-1.5 leading-none">
          <span className="text-xs font-medium text-ink-muted">Tokens</span>
          <span className="mt-0.5 text-[10px] italic tabular-nums text-ink-faint">
            {tokens === null || tokens === undefined ? "?" : tokens >= 1000 ? `${(tokens / 1000).toFixed(1).replace(/\.0$/, "")}k` : tokens}
          </span>
        </div>
      )}
      <StatsLine stats={stats} />
    </div>
  )
}

/** The session's live dot and folder, at the composer's left like omp-web's working directory. */
export function PiRepoLabel({ repo, tone }: { repo: string; tone?: StatusTone }) {
  return (
    <span className="flex min-w-0 items-center gap-2 px-1.5 text-xs">
      {tone && <StatusDot tone={tone} />}
      <span className="truncate font-medium text-ink-muted">{repo}</span>
    </span>
  )
}

/**
 * The compact header of a session in a side panel (a workbench chat): live dot and session name, then model, thinking, context and cost, then `actions`.
 *
 * Before the session runs it shows `controls` (what a new session starts on) instead.
 */
export function PiPanelBar({ header, controls, actions }: { header: PiSessionHeader | null; controls?: PiSessionControls; actions?: ReactNode }) {
  const shown = header?.controls ?? controls
  return (
    <div className="flex h-8 min-w-0 flex-1 items-center gap-1.5 px-2 text-xs">
      {header && <StatusDot tone={header.tone} />}
      {header?.sessionName && <span className="min-w-0 truncate text-ink-muted">{header.sessionName}</span>}
      <div className="ml-auto flex min-w-0 items-center gap-1">
        {shown && <PiSessionControlsBar controls={shown} compact />}
        {actions && <div className="flex shrink-0 items-center">{actions}</div>}
      </div>
    </div>
  )
}
