"use client"

import { useCallback, useState, type ReactNode } from "react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { Disclosure } from "@/components/ui/Disclosure"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { readMemory } from "@/lib/backend/api"
import type { FactoryTask } from "@/lib/domain/factory"
import { STATE_LABELS } from "./states"
import { descriptionPath, type TagGroup } from "./tagGroups"

/**
 * The group's description row, the document `plan/<path>.md` of its plan's memory_store rendered as markdown when
 * opened. Read as soon as it mounts (when its group opens), the row appears only once a non-empty document came back:
 * a missing document or memory_store (or any failed read) renders nothing, a plan need not describe every level.
 */
export function TagGroupDescription({ plan, path }: { plan: string; path: string }) {
  const [text, setText] = useState<string | null>(null)
  const load = useCallback(() => readMemory(plan, descriptionPath(path)), [plan, path])
  useLoadWhile(true, load, (read) => setText(read.kind === "document" && read.text.trim() ? read.text : null))
  if (text === null) return null
  return (
    <Disclosure summary="Description" summaryClassName="text-ink-subtle">
      <div className="py-1 text-xs text-ink-muted"><LaTeXMarkdown content={text} compact /></div>
    </Disclosure>
  )
}

function GroupSummary({ group }: { group: TagGroup }) {
  return (
    <span className="flex min-w-0 items-baseline gap-2">
      <span className="truncate">{group.name}</span>
      <span className="shrink-0 text-[11px] tabular-nums text-ink-faint">{group.count}</span>
      {group.states.length > 0 && (
        <span className="min-w-0 truncate text-[11px] text-ink-faint">
          {group.states.map(({ state, count }) => `${count} ${STATE_LABELS[state].toLowerCase()}`).join(" · ")}
        </span>
      )}
    </span>
  )
}

/** One group: its row, and while open its description (when its document exists), the groups below it and its own tasks as cards. */
function TagGroupNode({ group, renderTask }: { group: TagGroup; renderTask: (task: FactoryTask) => ReactNode }) {
  return (
    <Disclosure summary={<GroupSummary group={group} />}>
      <div className="flex flex-col pl-5">
        {group.plan != null && <TagGroupDescription plan={group.plan} path={group.path} />}
        {group.children.map((child) => <TagGroupNode key={child.path} group={child} renderTask={renderTask} />)}
        {group.tasks.length > 0 && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(15rem,1fr))] gap-2.5 py-2">
            {group.tasks.map((task) => <div key={task.id}>{renderTask(task)}</div>)}
          </div>
        )}
      </div>
    </Disclosure>
  )
}

/**
 * The plan tree above the board's columns: a row per tag segment (name, task count, a wave's counts per state), closed
 * until opened. `renderTask` draws a task as the columns do.
 */
export function TagTree({ groups, renderTask }: { groups: readonly TagGroup[]; renderTask: (task: FactoryTask) => ReactNode }) {
  return (
    <section aria-label="Plans" className="flex flex-col">
      {groups.map((group) => <TagGroupNode key={group.path} group={group} renderTask={renderTask} />)}
    </section>
  )
}
