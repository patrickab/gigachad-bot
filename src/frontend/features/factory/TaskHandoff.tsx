"use client"

import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import type { FactoryTask, FactoryTimelineEntry } from "@/lib/domain/factory"
import { formatCost, formatUsage } from "./usage"
import type { TaskDiffFile } from "./useTaskDiff"

/** What the coder said last: its handoff, undefined while it has said nothing. */
export function handoffOf(entries: readonly FactoryTimelineEntry[]): string | undefined {
  for (let index = entries.length - 1; index >= 0; index--) {
    const entry = entries[index]
    if (entry.kind === "coder" && entry.data.text.trim()) return entry.data.text
  }
  return undefined
}

/**
 * The coder's last message rendered as markdown, with the files it changed below: a file opens the diff.
 * `onShowChanges` switches the surface to the diff.
 */
export function TaskHandoff({ task, text, files, onShowChanges }: {
  task: FactoryTask
  text: string
  files: readonly TaskDiffFile[]
  onShowChanges: () => void
}) {
  const added = files.reduce((sum, file) => sum + (file.added ?? 0), 0)
  const removed = files.reduce((sum, file) => sum + (file.removed ?? 0), 0)
  return (
    <div className="h-full overflow-y-auto scrollbar-thin">
      <article aria-label="Coder handoff" className="mx-auto flex max-w-3xl flex-col gap-6 px-6 py-8">
        <div className="text-sm text-ink"><LaTeXMarkdown content={text} /></div>
        {files.length > 0 && (
          <section aria-label="Changed files" className="flex flex-col gap-1 text-xs">
            <div className="flex items-baseline gap-2 pb-1 text-ink-subtle">
              <span>{files.length} {files.length === 1 ? "file" : "files"} changed</span>
              <span className="tabular-nums text-ink-faint">+{added} −{removed}</span>
              <button type="button" onClick={onShowChanges} className="ml-auto text-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">View diff</button>
            </div>
            <ul className="flex flex-col">
              {files.map((file) => (
                <li key={file.path}>
                  <button type="button" onClick={onShowChanges} className="flex w-full items-baseline gap-3 rounded px-2 py-1 text-left hover:bg-surface-elevated focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
                    <span className="min-w-0 flex-1 truncate font-mono text-ink-muted">{file.path}</span>
                    <span className="shrink-0 tabular-nums text-ink-faint">+{file.added ?? 0} −{file.removed ?? 0}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}
        {task.usage && <p className="text-[11px] tabular-nums text-ink-faint">{formatUsage(task.usage)} · {formatCost(task.usage.cost)}</p>}
      </article>
    </div>
  )
}
