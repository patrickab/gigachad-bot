"use client"

/** Compact token readout for the top bar. Hidden until the first tokens are counted. */
export function TokenCounter({ tokens }: { tokens: number }) {
  if (tokens === 0) return null

  const fmt = (n: number) =>
    n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)

  return (
    <div className="flex flex-col items-center px-2 py-1 tabular-nums">
      <span className="text-sm font-medium text-ink-muted leading-none">Tokens</span>
      <span className="text-[10px] italic text-ink-subtle leading-none mt-0.5">{fmt(tokens)}</span>
    </div>
  )
}
