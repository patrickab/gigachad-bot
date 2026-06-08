"use client"

import type { Usage } from "@/lib/types"

interface TokenCounterProps {
  usage: Usage
}

export function TokenCounter({ usage }: TokenCounterProps) {
  if (usage.total_tokens === 0) return null

  const fmt = (n: number) =>
    n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)

  return (
    <div className="flex flex-col items-center px-2 py-1 tabular-nums" title={`Input: ${usage.prompt_tokens} · Output: ${usage.completion_tokens} · Total: ${usage.total_tokens}`}>
      <span className="text-sm font-medium text-ink-muted leading-none">Tokens</span>
      <span className="text-[10px] italic text-ink-subtle leading-none mt-0.5">{fmt(usage.total_tokens)}</span>
    </div>
  )
}
