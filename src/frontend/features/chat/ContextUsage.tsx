"use client"

import { useResource } from "@/hooks/useResource"
import { piCatalogResource } from "@/lib/backend/resources"
import type { Usage } from "@/lib/domain/types"
import { gatewayModelId } from "@/lib/models"

export function compactTokens(tokens: number): string {
  return tokens >= 1000 ? `${(tokens / 1000).toFixed(1).replace(/\.0$/, "")}k` : String(tokens)
}

/** A chat's readout, `12.3k | 1%`: the last turn's tokens against the selected model's context window, tokens only when it is unknown. Hidden before the first turn. */
export function ChatContextUsage({ usage, model }: { usage: Usage; model: string }) {
  const catalog = useResource(null, piCatalogResource)
  if (!usage.total_tokens) return null
  const gatewayId = gatewayModelId(model)
  const contextWindow = catalog?.providers.flatMap((provider) => provider.models).find((entry) => entry.id === gatewayId)?.context_window
  const percent = contextWindow ? (usage.total_tokens / contextWindow) * 100 : undefined
  const tokensText = compactTokens(usage.total_tokens)
  const percentText = percent === undefined ? null : `${Math.round(percent)}%`
  const label = `Context ${tokensText} tokens${percentText ? `, ${percentText}` : ""}${contextWindow ? ` of ${compactTokens(contextWindow)}` : ""}`
  return (
    <span aria-label={label} className="shrink-0 whitespace-nowrap px-3 font-mono text-xs tabular-nums text-ink-faint">
      {tokensText}{percentText && ` | ${percentText}`}
    </span>
  )
}
