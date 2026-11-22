// Token and dollar figures of a factory task, shared by its timeline and its board card.

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
