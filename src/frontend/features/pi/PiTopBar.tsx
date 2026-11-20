"use client"

import type { ComponentProps, ReactNode } from "react"
import { StatusDot, type StatusTone } from "@/components/ui/StatusDot"
import { compactTokens, ContextUsage } from "@/features/chat/ContextUsage"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import { catalogOffering, gatewayModelId, selectorIdOf } from "@/lib/models"
import { PiModelSelector, PiThinkingSelector } from "./PiModelSelector"
import type { PiModel } from "./types"
import type { PiSessionStats } from "./usePiSession"

/** The session's model and thinking level as a header shows and changes them, with every model Pi offers it (`get_available_models`). */
export interface PiModelControls {
  model: PiModel | null
  models: PiModel[]
  thinkingLevel: string | null
  /** Levels the model offers, `["off"]` (no selector) for a model without reasoning. */
  thinkingLevels: string[]
  onModel: (model: PiModel) => void
  onThinking: (level: string) => void
}

/** The session's model, thinking and usage. */
export interface PiSessionControls extends PiModelControls {
  stats: PiSessionStats | null
}

/** The app's model dropdown as chats get it: the user's catalog, its saves, Manage models and default models. */
export type PiCatalogProps = Omit<ComponentProps<typeof ModelDropdown>, "selectedModel" | "onSelect" | "pickModels" | "chatColumn">

/** Pi's `<provider>/<id>` of a model, how the gateway and the catalog's selectors (`gatewayModelId`) name it. */
export function piGatewayId(model: PiModel): string {
  return `${model.provider}/${model.id}`
}

/** The Pi model a catalog selector names (`openrouter/acme/x` -> provider `openrouter`, id `acme/x`), null for one without a provider. */
export function piModelOfSelector(selector: string): PiModel | null {
  const gatewayId = gatewayModelId(selector)
  const slash = gatewayId.indexOf("/")
  return slash > 0 && slash < gatewayId.length - 1 ? { provider: gatewayId.slice(0, slash), id: gatewayId.slice(slash + 1) } : null
}

/** What a session view's header shows. */
export interface PiSessionHeader {
  tone: StatusTone
  sessionName: string | undefined
  controls: PiSessionControls
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

function ThinkingControl({ controls }: { controls: PiModelControls }) {
  return controls.thinkingLevel && controls.thinkingLevels.length > 1 && (
    <div className="shrink-0">
      <PiThinkingSelector levels={controls.thinkingLevels} selected={controls.thinkingLevel} onSelect={controls.onThinking} />
    </div>
  )
}

/** Model and thinking selectors, then the usage line, for a side panel's session header. `compact` shortens the model name. */
export function PiSessionControlsBar({ controls, compact = false }: { controls: PiSessionControls; compact?: boolean }) {
  return (
    <div className={compact ? "flex min-w-0 items-center gap-1" : "flex shrink-0 items-center gap-1"}>
      {controls.models.length > 0 && <PiModelSelector models={controls.models} selected={controls.model} onSelect={controls.onModel} compact={compact} />}
      <ThinkingControl controls={controls} />
      <StatsLine stats={controls.stats} />
    </div>
  )
}

/**
 * A Pi tab's top bar: session name, then the session's model and thinking once it runs, or `controls` (what a new session starts on) before.
 *
 * The model is the app's dropdown over the user's catalog (`modelProps`): it offers the catalog models Pi offers the session, shows the session's
 * model by its catalog selector, and opens Manage models and default models like a chat's. Hidden until Pi names its models.
 */
export function PiTopBar({ header, controls, modelProps }: { header: PiSessionHeader | null; controls?: PiModelControls; modelProps: PiCatalogProps }) {
  const shown = header?.controls ?? controls
  const current = shown?.model ? piGatewayId(shown.model) : ""
  const select = (selector: string) => {
    const model = piModelOfSelector(selector)
    if (model) shown?.onModel(model)
  }
  return (
    <div className="flex min-w-0 flex-1 items-center gap-2 px-2 text-sm">
      {header?.sessionName && <span className="truncate text-ink-muted">{header.sessionName}</span>}
      {shown && (
        <div className="flex shrink-0 items-center gap-1">
          {shown.models.length > 0 && (
            <ModelDropdown
              {...modelProps}
              pickModels={(catalog) => catalogOffering(catalog, new Set(shown.models.map(piGatewayId)))}
              selectedModel={current && modelProps.models ? selectorIdOf(modelProps.models, current) : current}
              onSelect={select}
            />
          )}
          <ThinkingControl controls={shown} />
        </div>
      )}
    </div>
  )
}

/** The top bar's right end: context tokens and context use (F3), the same readout as a chat's. Nothing before Pi reports them. */
export function PiUsageBar({ stats }: { stats: PiSessionStats | null }) {
  const usage = stats?.contextUsage
  if (!usage) return null
  return <ContextUsage tokens={usage.tokens} percent={usage.percent} contextWindow={usage.contextWindow} />
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
