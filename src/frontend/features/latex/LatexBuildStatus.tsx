"use client"

import { MenuPopover } from "@/components/ui/MenuPopover"
import type { LatexBuildState } from "./useLatexBuild"

/** What the top bar says about the build, null before the first build ran. `problem` when the text opens the details. */
function statusOf({ building, result, failure }: LatexBuildState): { text: string; problem: boolean } | null {
  if (building) return { text: "Building…", problem: false }
  if (failure) return { text: failure.notInstalled ? "LaTeX not installed" : "Build failed", problem: true }
  if (!result) return null
  if (!result.ok) return { text: "Build failed", problem: true }
  return { text: `Built ${(result.duration_ms / 1000).toFixed(1)}s`, problem: false }
}

/** Build status in the top bar: a dot and a few words. A failure opens the parsed TeX errors and the log tail. */
export function LatexBuildStatus({ build }: { build: LatexBuildState }) {
  const status = statusOf(build)
  if (!status) return null
  const dot = <span aria-hidden className={`h-1.5 w-1.5 shrink-0 rounded-full ${status.problem ? "bg-danger" : "bg-ink-faint"}`} />
  const label = <span className="truncate">{status.text}</span>
  const rowClass = "flex items-center gap-1.5 px-1 text-[11px] text-ink-subtle"
  if (!status.problem) return <div role="status" className={rowClass}>{dot}{label}</div>

  return (
    <MenuPopover
      trigger={(open, toggle) => (
        <button type="button" aria-expanded={open} onClick={toggle} className={`${rowClass} rounded transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted`}>
          {dot}{label}
        </button>
      )}
    >
      {(open) => open && (
        <div className="absolute right-0 top-full z-50 mt-2 flex max-h-[60vh] w-[28rem] max-w-[80vw] flex-col gap-2 overflow-y-auto rounded-2xl border border-divider/50 bg-surface p-3 text-xs shadow-[var(--shadow-lg)]">
          {build.failure && <p className="text-danger">{build.failure.message}</p>}
          {!build.failure && build.result?.errors.map((error, index) => (
            <p key={index} className="text-ink">
              {error.line !== null && <span className="mr-2 text-ink-subtle">line {error.line}</span>}
              {error.message}
            </p>
          ))}
          {!build.failure && build.result?.log_tail && <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-[11px] text-ink-subtle">{build.result.log_tail}</pre>}
        </div>
      )}
    </MenuPopover>
  )
}
