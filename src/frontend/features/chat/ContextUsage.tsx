"use client"

import { useResource } from "@/hooks/useResource"
import { piCatalogResource } from "@/lib/backend/resources"
import type { Usage } from "@/lib/domain/types"
import { gatewayModelId } from "@/lib/models"

export function compactTokens(tokens: number): string {
  return tokens >= 1000 ? `${(tokens / 1000).toFixed(1).replace(/\.0$/, "")}k` : String(tokens)
}

/**
 * The top bar's context readout, `12.3k | 1%`, the same for chats and Pi tabs.
 *
 * - `tokens` / `percent`: null shows `?` (unknown, e.g. right after a compaction)
 * - `percent` undefined (context window unknown): tokens only
 */
export function ContextUsage({ tokens, percent, contextWindow }: { tokens: number | null; percent?: number | null; contextWindow?: number | null }) {
  const tokensText = tokens === null ? "?" : compactTokens(tokens)
  const percentText = percent === undefined ? null : percent === null ? "?" : `${Math.round(percent)}%`
  const label = `Context ${tokensText} tokens${percentText ? `, ${percentText}` : ""}${contextWindow ? ` of ${compactTokens(contextWindow)}` : ""}`
  return (
    <span aria-label={label} className="shrink-0 whitespace-nowrap px-3 font-mono text-xs tabular-nums text-ink-faint">
      {tokensText}{percentText && ` | ${percentText}`}
    </span>
  )
}

/** A chat's readout: the last turn's tokens against the selected model's context window. Hidden before the first turn. */
export function ChatContextUsage({ usage, model }: { usage: Usage; model: string }) {
  const catalog = useResource(null, piCatalogResource)
  if (!usage.total_tokens) return null
  const gatewayId = gatewayModelId(model)
  const contextWindow = catalog?.providers.flatMap((provider) => provider.models).find((entry) => entry.id === gatewayId)?.context_window
  const percent = contextWindow ? (usage.total_tokens / contextWindow) * 100 : undefined
  return <ContextUsage tokens={usage.total_tokens} percent={percent} contextWindow={contextWindow} />
}
