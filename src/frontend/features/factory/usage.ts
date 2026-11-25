// Token and dollar figures of a factory task or wave, shared by the timeline, the board's cards and its wave containers.

/** Dollars, to the tenth of a cent below ten cents: a review on a small model costs less than one cent. */
export function formatCost(cost: number): string {
  return `$${cost.toFixed(cost < 0.1 ? 3 : 2)}`
}

/** Token count, compact from a thousand on: 950, 1.2k, 48k, 3.1M. */
export function formatTokens(count: number): string {
  if (count < 1000) return String(count)
  if (count < 1_000_000) return `${(count / 1000).toFixed(count < 10_000 ? 1 : 0)}k`
  return `${(count / 1_000_000).toFixed(1)}M`
}

export function formatUsage(usage: { input: number; output: number }): string {
  return `${formatTokens(usage.input)} in / ${formatTokens(usage.output)} out`
}

/** Tokens in and out together, then dollars: a card's and a wave row's spend. */
export function formatSpend(usage: { input: number; output: number; cost: number }): string {
  return `${formatTokens(usage.input + usage.output)} tok · ${formatCost(usage.cost)}`
}

/** Flow and spend of a wave header: `↑55k ↓12k · $3.50`, in = input plus cache reads, out = output. */
export function formatFlow(usage: { input: number; output: number; cost: number; cacheRead?: number }): string {
  return `↑${formatTokens(usage.input + (usage.cacheRead ?? 0))} ↓${formatTokens(usage.output)} · ${formatCost(usage.cost)}`
}
