"use client"

import { LoaderCircle } from "lucide-react"
import type { ComponentProps, ReactNode } from "react"
import { createPortal } from "react-dom"
import { StatusDot, type StatusTone } from "@/components/ui/StatusDot"
import { compactTokens } from "@/features/chat/ContextUsage"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import { ReasoningSelector } from "@/features/settings/ReasoningSelector"
import { catalogOffering, gatewayModelId, selectorIdOf } from "@/lib/models"
import { baseName } from "@/lib/paths"
import type { PiModel } from "./types"
import type { PiSessionStats } from "./protocol"

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

/**
 * The gateway's `<provider>/<id>` of a Pi model, how the catalog's selectors (`gatewayModelId`) name it. A model of Pi's provider `gateway`
 * (pi-host's gateway, for providers the session has no key for) is already named so: `gateway` + `openrouter/acme/x` is `openrouter/acme/x`.
 */
export function piGatewayId(model: PiModel): string {
  return model.provider === "gateway" ? model.id : `${model.provider}/${model.id}`
}

/** The model of `models` a catalog selector names, the session's own provider before `gateway`; null when the session offers none. */
export function piModelFor(models: PiModel[], selector: string): PiModel | null {
  const gatewayId = gatewayModelId(selector)
  const matches = models.filter((model) => piGatewayId(model) === gatewayId)
  return matches.find((model) => model.provider !== "gateway") ?? matches[0] ?? null
}

/** What a session view's header shows. */
export interface PiSessionHeader {
  tone: StatusTone
  sessionName: string | undefined
  controls: PiSessionControls
  /** The transcript's turn column, which the model dropdown's name makes way for like a chat's. */
  column: HTMLElement | null
}

/** The session's thinking level: the chats' reasoning selector over the model's ladder. Nothing for a model without reasoning (`["off"]`). */
export function PiThinkingSelector({ controls }: { controls: PiModelControls }) {
  const levels = controls.thinkingLevel && controls.thinkingLevels.length > 1 ? controls.thinkingLevels : []
  return <ReasoningSelector levels={levels} reasoningEffort={controls.thinkingLevel ?? ""} onReasoningChange={controls.onThinking} placement="below" />
}

/**
 * The session's model: the app's dropdown over the user's catalog (`modelProps`), as a chat's.
 *
 * It offers the catalog models Pi offers the session, shows the session's model by its catalog selector, and opens Manage models and default
 * models like a chat's. Its name collapses where it would cover `chatColumn`. Nothing until Pi names its models.
 */
export function PiModelDropdown({ controls, modelProps, chatColumn }: { controls: PiModelControls; modelProps: PiCatalogProps; chatColumn?: HTMLElement | null }) {
  if (controls.models.length === 0) return null
  const current = controls.model ? piGatewayId(controls.model) : ""
  const select = (selector: string) => {
    const model = piModelFor(controls.models, selector)
    if (model) controls.onModel(model)
  }
  return (
    <ModelDropdown
      {...modelProps}
      pickModels={(catalog) => catalogOffering(catalog, new Set(controls.models.map(piGatewayId)))}
      selectedModel={current && modelProps.models ? selectorIdOf(modelProps.models, current) : current}
      onSelect={select}
      chatColumn={chatColumn}
    />
  )
}

/** `0.16$ (0.40$)`: the main agent's cost, then its subagents' and factory tasks' in brackets. */
export function formatSessionCost(cost: number, subCost: number): string {
  return `${cost.toFixed(2)}$ (${subCost.toFixed(2)}$)`
}

/** Context use colour: muted to 20%, then highlighted, yellow from 40%, red from 60%. */
export function contextTone(percent: number | null): string {
  if (percent === null || percent < 20) return "text-ink-faint"
  if (percent < 40) return "text-ink"
  return percent < 60 ? "text-warning" : "text-danger"
}

/** Share of the session's prompt tokens read from cache, null before any. */
export function cacheHitRate(tokens: PiSessionStats["tokens"]): number | null {
  if (!tokens) return null
  const prompt = tokens.input + tokens.cacheRead + tokens.cacheWrite
  return prompt > 0 ? tokens.cacheRead / prompt : null
}

const Dot = () => <span aria-hidden="true" className="px-1.5 text-ink-subtle">·</span>

/**
 * The telemetry readout as one line, `⟳2 · 0.48$ (0.00$) · 62.8k (6%) · 82% cached`: working subsessions (only while some work), session
 * cost, context tokens and use (F3) coloured by `contextTone`, cache hit rate. Nothing before Pi reports them.
 */
export function PiUsage({ stats }: { stats: PiSessionStats | null }) {
  if (!stats) return null
  const usage = stats.contextUsage
  const active = stats.subActive ?? 0
  const cost = formatSessionCost(stats.cost, stats.subCost ?? 0)
  const tokens = usage ? (usage.tokens === null ? "?" : compactTokens(usage.tokens)) : null
  const percent = usage ? (usage.percent === null ? "?" : `${Math.round(usage.percent)}%`) : null
  const rate = cacheHitRate(stats.tokens)
  const cached = rate === null ? null : `${Math.round(rate * 100)}% cached`
  const label = [
    active > 0 && `${active} subsessions working`,
    `Cost ${cost}`,
    usage && `context ${tokens} tokens, ${percent} of ${compactTokens(usage.contextWindow)}`,
    cached,
  ].filter(Boolean).join(", ")
  return (
    <span aria-label={label} className="flex shrink-0 items-center whitespace-nowrap px-3 font-mono text-xs tabular-nums text-ink-faint">
      {active > 0 && <><span className="flex items-center gap-1"><LoaderCircle size={11} aria-hidden="true" className="animate-spin" />{active}</span><Dot /></>}
      {cost}
      {usage && <><Dot /><span className={contextTone(usage.percent)}>{tokens} ({percent})</span></>}
      {cached && <><Dot />{cached}</>}
    </span>
  )
}

/** The tab top bar's places a Pi tab fills, where a chat's bar has its reasoning selector, model dropdown and context readout. */
export interface PiTopBarSlots {
  thinking: HTMLElement | null
  model: HTMLElement | null
  usage: HTMLElement | null
}

/**
 * A Pi tab's controls in the tab's top bar, placed like a chat's: thinking, then the model and the session name, the context readout at the right.
 *
 * They show the running session's (`header`), or before it runs `controls`, what a new session starts on.
 */
export function PiTopBar({ slots, header, controls, modelProps }: { slots: PiTopBarSlots; header: PiSessionHeader | null; controls?: PiModelControls; modelProps: PiCatalogProps }) {
  const shown = header?.controls ?? controls
  return <>
    {slots.thinking && shown && createPortal(<PiThinkingSelector controls={shown} />, slots.thinking)}
    {slots.model && createPortal(<>
      {shown && <PiModelDropdown controls={shown} modelProps={modelProps} chatColumn={header?.column} />}
      {header?.sessionName && <span className="ml-2 truncate text-sm text-ink-muted">{header.sessionName}</span>}
    </>, slots.model)}
    {slots.usage && header && createPortal(<PiUsage stats={header.controls.stats} />, slots.usage)}
  </>
}

/** The session's live dot and folder at the composer's left, where an empty chat shows the project it starts a session in. */
/** The mount a composer runs in, after its state: a `tone` dot, or `mark` (e.g. the session's live dot) when given. */
export function PiRepoLabel({ repo, tone, mark }: { repo: string; tone?: StatusTone; mark?: ReactNode }) {
  return (
    <span className="flex h-8 min-w-0 items-center gap-1.5 px-2 text-xs text-ink">
      {mark ?? (tone && <StatusDot tone={tone} />)}
      <span className="truncate">{baseName(repo)}</span>
    </span>
  )
}

/**
 * The header of a session in a side panel (a workbench chat), ordered like a chat's top bar: live dot, thinking, model and session name, then
 * the context readout and `actions`.
 */
export function PiPanelBar({ header, modelProps, actions }: { header: PiSessionHeader | null; modelProps: PiCatalogProps; actions?: ReactNode }) {
  const shown = header?.controls
  return (
    <div className="flex min-w-0 flex-1 items-center gap-1">
      {header && <span className="flex shrink-0 px-1"><StatusDot tone={header.tone} /></span>}
      {shown && <PiThinkingSelector controls={shown} />}
      {shown && <PiModelDropdown controls={shown} modelProps={modelProps} />}
      {header?.sessionName && <span className="min-w-0 truncate text-xs text-ink-muted">{header.sessionName}</span>}
      <div className="ml-auto flex shrink-0 items-center">
        <PiUsage stats={header?.controls.stats ?? null} />
        {actions}
      </div>
    </div>
  )
}
