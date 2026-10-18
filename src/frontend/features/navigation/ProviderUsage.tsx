"use client"

import { useCallback, useEffect, useState, useSyncExternalStore } from "react"
import { RefreshCw } from "lucide-react"
import { cn } from "@/lib/utils"
import { fetchProviderUsage, type ProviderUsageReport, type ProviderUsageWindow } from "@/lib/backend/api"

const REFRESH_INTERVAL_MS = 5 * 60_000
const OPEN_STORAGE_KEY = "provider_usage_open"
const HIGH_USAGE_PERCENT = 80

export function formatReset(minutes: number): string {
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return minutes % 60 ? `${hours}h ${minutes % 60}m` : `${hours}h`
  const days = Math.floor(hours / 24)
  return hours % 24 ? `${days}d ${hours % 24}h` : `${days}d`
}

function worstWindow(report: ProviderUsageReport): ProviderUsageWindow | null {
  return report.windows.reduce<ProviderUsageWindow | null>((worst, window) => (!worst || window.percent > worst.percent ? window : worst), null)
}

const isHigh = (percent: number) => percent >= HIGH_USAGE_PERCENT

function Meter({ percent }: { percent: number }) {
  return <div className="h-0.5 overflow-hidden rounded-full bg-divider">
    <div className={cn("h-full rounded-full", isHigh(percent) ? "bg-danger" : "bg-ink-faint")} style={{ width: `${Math.min(100, percent)}%` }} />
  </div>
}

function AccountRow({ report }: { report: ProviderUsageReport }) {
  const [open, setOpen] = useState(false)
  if (report.unavailable) return <div className="flex items-baseline gap-2 px-2 pt-1 pb-1.5 text-xs">
    <span className="shrink-0 text-ink">{report.provider}</span>
    <span className="min-w-0 flex-1 truncate text-ink-faint">{report.account}</span>
    <span className="shrink-0 text-ink-faint">unavailable</span>
  </div>
  const worst = worstWindow(report)
  const percent = Math.round(worst?.percent ?? 0)
  return <div>
    <button
      onClick={() => setOpen((value) => !value)} aria-expanded={open}
      aria-label={`${report.provider} ${report.account ?? ""}`.trim()}
      className="flex w-full items-baseline gap-2 rounded-md px-2 pt-1 text-left text-xs hover:bg-hover focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
    >
      <span className="shrink-0 text-ink">{report.provider}</span>
      <span className="min-w-0 flex-1 truncate text-ink-faint">{report.account}</span>
      <span className={cn("shrink-0 tabular-nums", isHigh(percent) ? "text-danger" : "text-ink-subtle")}>{percent}%</span>
    </button>
    <div className="mx-2 mb-1 mt-1"><Meter percent={worst?.percent ?? 0} /></div>
    {open && <ul className="mb-1 space-y-1 px-2">
      {report.windows.map((window) => <li key={window.id} className="flex items-center gap-2 text-[10px] text-ink-faint">
        <span className="w-5 shrink-0 uppercase">{window.id}</span>
        <div className="flex-1"><Meter percent={window.percent} /></div>
        <span className="w-7 shrink-0 text-right tabular-nums">{Math.round(window.percent)}%</span>
        <span className="w-14 shrink-0 text-right">{window.resets_in_minutes != null && `in ${formatReset(window.resets_in_minutes)}`}</span>
      </li>)}
    </ul>}
  </div>
}

const OPEN_EVENT = "provider-usage-open"

function subscribeOpen(callback: () => void) {
  window.addEventListener(OPEN_EVENT, callback)
  return () => window.removeEventListener(OPEN_EVENT, callback)
}

/** Subscription limits of every OMP login: a one-line summary that opens into one meter per account. */
export function ProviderUsage() {
  const [reports, setReports] = useState<ProviderUsageReport[] | null>(null)
  const [failed, setFailed] = useState(false)
  const [loading, setLoading] = useState(false)
  const open = useSyncExternalStore(subscribeOpen, () => localStorage.getItem(OPEN_STORAGE_KEY) === "true", () => false)

  const load = useCallback((forced: boolean) => fetchProviderUsage(forced).then(
    (usage) => { setReports(usage.reports.filter((report) => !report.no_limits)); setFailed(false) },
    () => { setFailed(true) },
  ), [])
  const refresh = async () => {
    setLoading(true)
    await load(true)
    setLoading(false)
  }
  useEffect(() => {
    void load(false)
    const interval = window.setInterval(() => void load(false), REFRESH_INTERVAL_MS)
    return () => window.clearInterval(interval)
  }, [load])

  if (!failed && reports?.length === 0) return null
  // Headroom: each account counts with its binding window, the header shows the least used account.
  const worst = (reports ?? []).map(worstWindow).reduce<ProviderUsageWindow | null>((least, window) => (window && (!least || window.percent < least.percent) ? window : least), null)
  const summary = failed ? "unavailable" : worst ? `${Math.round(worst.percent)}% · ${worst.id}` : "…"

  return <section aria-label="Provider usage" className="group/usage mx-2 shrink-0 border-t border-divider/50 py-2">
    <div className="flex h-6 items-center justify-between gap-1 px-2">
      <button
        aria-expanded={open}
        onClick={() => { localStorage.setItem(OPEN_STORAGE_KEY, String(!open)); window.dispatchEvent(new Event(OPEN_EVENT)) }}
        className="flex min-w-0 flex-1 items-center justify-between gap-2 text-xs text-ink-subtle hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
      >
        <span className="font-medium">Usage</span>
        <span className={cn("truncate tabular-nums", worst && isHigh(worst.percent) ? "text-danger" : "text-ink-faint")}>{summary}</span>
      </button>
      <button
        aria-label="Refresh usage" disabled={loading} onClick={() => void refresh()}
        className="shrink-0 rounded p-1 text-ink-faint opacity-0 transition-opacity hover:text-ink focus-visible:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted group-hover/usage:opacity-100 disabled:opacity-100"
      >
        <RefreshCw className={cn("h-3 w-3", loading && "animate-spin motion-reduce:animate-none")} />
      </button>
    </div>
    {open && <div className="mt-1 max-h-48 space-y-1 overflow-y-auto">{(reports ?? []).map((report, index) => <AccountRow key={`${report.provider}:${report.account ?? index}`} report={report} />)}</div>}
  </section>
}
