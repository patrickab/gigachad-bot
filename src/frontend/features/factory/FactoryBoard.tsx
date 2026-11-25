"use client"

import { useCallback, useMemo, useState } from "react"
import { useTabWorkingDirectory } from "@/contexts/tabWorkingDirectory"
import { motion } from "framer-motion"
import { Archive, ArchiveRestore, Plus, RotateCcw, Settings, Trash2, X } from "lucide-react"
import { BoardColumn } from "@/components/ui/BoardColumn"
import { ElevatedContainer, ElevationProvider } from "@/components/ui/ElevatedContainer"
import { ErrorText } from "@/components/ui/ErrorText"
import { IconButton } from "@/components/ui/IconButton"
import { InlineConfirm } from "@/components/ui/InlineConfirm"
import { StatusDot } from "@/components/ui/StatusDot"
import { InTopBar } from "@/features/navigation/TopBarSlots"
import { useStoredJson } from "@/hooks/useStoredJson"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { ApiError, callFactory } from "@/lib/backend/api"
import type { FactoryConfig, FactoryTask, FactoryTaskRemoval } from "@/lib/domain/factory"
import { baseName } from "@/lib/paths"
import { cn, errorMessage } from "@/lib/utils"
import { ConfigDraft } from "./ConfigDraft"
import { FactorySettingsDialog } from "./FactorySettingsDialog"
import { NewTaskDialog } from "./NewTaskDialog"
import { BOARD_COLUMNS, STATE_LABELS, shownState, taskStatusLine, taskTone } from "./states"
import { groupByTag } from "./tagGroups"
import { TagTree } from "./TagTree"
import { dismissLaunch, retryLaunch, useFactoryLaunches, type FactoryLaunch } from "./useFactoryTasks"
import { ICON_BUTTON_CLASS, PRIMARY_BUTTON_CLASS } from "./styles"
import { formatCost, formatTokens } from "./usage"

/** Finished tasks shown before "Show all", the column otherwise grows without bound. */
const FINISHED_LIMIT = 10

/** Column order of two tasks: the newest first. */
function boardOrder(left: FactoryTask, right: FactoryTask): number {
  return right.createdAt.localeCompare(left.createdAt)
}

/** Age of a task as the card shows it: minutes under an hour, hours under a day, days under a week, then weeks. */
export function formatAge(createdAt: string, now = Date.now()): string {
  const minutes = Math.max(0, Math.floor((now - Date.parse(createdAt)) / 60_000))
  if (minutes < 60) return `${minutes}min`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  return days < 7 ? `${days}d` : `${Math.floor(days / 7)}w`
}

/** The ids in `raw`, empty when it is not a list of strings. */
function parseArchived(raw: unknown): string[] {
  return Array.isArray(raw) ? raw.filter((id): id is string => typeof id === "string") : []
}

/** Whether `cause` is the core's refusal of a removal that `force` would allow. */
function forceWouldAllow(cause: unknown): boolean {
  return cause instanceof ApiError && cause.code === "refused" && cause.detail?.force === true
}

/**
 * One task on the board. Delete shows while the core accepts `tasks.remove` and asks first; a refusal that `force`
 * would allow (unmerged work) then shows with a second question that deletes it anyway, any other shows its message.
 */
function TaskCard({ task, archived, onArchive, onOpen, onDelete, onDeleteError }: {
  task: FactoryTask
  /** Whether the task is archived, null when it cannot be (not finished). */
  archived: boolean | null
  onArchive: () => void
  onOpen: () => void
  /** Delete the task, dropping its unmerged work with `force`. Throws the core's refusal. */
  onDelete: (force: boolean) => Promise<unknown>
  onDeleteError: (message: string) => void
}) {
  const [confirming, setConfirming] = useState(false)
  const [deleting, setDeleting] = useState(false)
  /** The core's refusal of a delete without `force`, null until one. */
  const [refusal, setRefusal] = useState<string | null>(null)
  const state = shownState(task)
  const failed = state === "blocked" || state === "interrupted"
  const close = () => {
    setConfirming(false)
    setRefusal(null)
  }
  const remove = async (force: boolean) => {
    setDeleting(true)
    onDeleteError("")
    try {
      await onDelete(force)
      close()
    } catch (cause) {
      if (!force && forceWouldAllow(cause)) setRefusal(errorMessage(cause, "Could not delete the task"))
      else {
        onDeleteError(errorMessage(cause, "Could not delete the task"))
        close()
      }
    } finally {
      setDeleting(false)
    }
  }
  return (
    <ElevatedContainer hoverLift className="group relative rounded-xl border border-divider/40 shadow-[var(--shadow-md)]">
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full flex-col gap-2 rounded-xl p-4 text-left focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
      >
        <span className="line-clamp-2 pr-6 text-sm font-semibold text-ink">{task.title}</span>
        <span className="flex min-w-0 items-center gap-2 text-[11px]">
          <StatusDot tone={taskTone(task)} />
          <span className={cn("min-w-0 flex-1 truncate", failed ? "text-danger" : "text-ink-subtle")}>{taskStatusLine(task)}</span>
          <span aria-label="Age" className="shrink-0 tabular-nums text-ink-faint">{formatAge(task.createdAt)}</span>
        </span>
        {task.kind === "merge_request" && (
          <span aria-label="Merge request" className="truncate text-[11px] text-ink-subtle">merge request · <code className="font-mono">{task.branch}</code></span>
        )}
        {task.settings?.models.coder && (
          <span aria-label="Coder model" className="truncate text-[11px] text-ink-faint">{task.settings.models.coder}</span>
        )}
        {state === "blocked" && task.reason && <span className="truncate text-[11px] text-ink-subtle">{task.reason}</span>}
        {state === "waiting_answer" && task.question && <span aria-label="Question" className="line-clamp-3 whitespace-pre-wrap text-[11px] text-ink-muted">{task.question}</span>}
        {refusal !== null && <span className="line-clamp-3 text-[11px] text-ink-subtle">{refusal}</span>}
        {/* The delete question covers this row, so asking never resizes the card. */}
        <span className={cn("flex h-6 min-w-0 items-center gap-1 text-[11px] text-ink-faint", confirming && "invisible")}>
          {task.usage && (
            <span aria-label="Tokens and cost" className="ml-auto shrink-0 tabular-nums">
              {formatTokens(task.usage.input + task.usage.output)} tok · {formatCost(task.usage.cost)}
            </span>
          )}
        </span>
      </button>
      {!confirming && (
        <div className="absolute right-2 top-2 flex opacity-0 focus-within:opacity-100 group-hover:opacity-100">
          {archived !== null && (
            <IconButton
              icon={archived ? ArchiveRestore : Archive}
              label={`${archived ? "Unarchive" : "Archive"} ${task.title}`}
              onClick={onArchive}
            />
          )}
          {task.actions.includes("tasks.remove") && (
            <IconButton icon={Trash2} label={`Delete ${task.title}`} onClick={() => setConfirming(true)} />
          )}
        </div>
      )}
      {confirming && (
        <div className="absolute inset-x-4 bottom-4 grid h-6 items-center">
          {refusal === null
            ? <InlineConfirm question={task.state === "done" && task.reason !== "no_changes" ? "Delete and revert its code?" : "Delete this task?"} busy={deleting} onYes={() => void remove(false)} onNo={close} />
            : <InlineConfirm question="Delete anyway?" busy={deleting} onYes={() => void remove(true)} onNo={close} />}
        </div>
      )}
    </ElevatedContainer>
  )
}

/** A start on its way, "Launching session" in Working, or refused, with its reason, in Waiting. */
function LaunchCard({ launch }: { launch: FactoryLaunch }) {
  const { repo, title } = launch.body
  const failed = launch.error !== null
  return (
    <ElevatedContainer className="relative rounded-xl border border-divider/40 shadow-[var(--shadow-md)]">
      <div role="group" aria-label={title} className="flex w-full flex-col gap-2 rounded-xl p-4">
        <span className="line-clamp-2 pr-14 text-sm font-semibold text-ink">{title}</span>
        <span role="status" className="flex min-w-0 items-center gap-2 text-[11px]">
          <StatusDot tone={failed ? "error" : "active"} />
          <span className={cn("truncate", failed ? "text-danger" : "text-ink-subtle")}>{failed ? "Could not start" : STATE_LABELS.starting}</span>
        </span>
        {failed && <span className="line-clamp-3 text-[11px] text-ink-subtle">{launch.error}</span>}
      </div>
      {failed && (
        <div className="absolute right-2 top-2 flex">
          <IconButton icon={RotateCcw} label={`Retry ${title}`} onClick={() => void retryLaunch(repo, launch.id)} />
          <IconButton icon={X} label={`Dismiss ${title}`} onClick={() => dismissLaunch(repo, launch.id)} />
        </div>
      )}
    </ElevatedContainer>
  )
}

/**
 * Kanban board of the tasks of the repo at path `repo`: what waits on the user first, then what runs, then what is
 * finished. Tasks tagged with a plan path leave the columns for the plan tree above them (`TagTree`).
 */
export function FactoryBoard({ repo, tasks, error, refresh, onOpenTask }: {
  repo: string
  tasks: FactoryTask[]
  /** Why the last read of the tasks failed, null when it did not. */
  error: string | null
  /** Read the tasks again, after a change this board made. */
  refresh: () => Promise<void>
  onOpenTask: (taskId: string) => void
}) {
  useTabWorkingDirectory(repo)
  const [newTaskOpen, setNewTaskOpen] = useState(false)
  const launches = useFactoryLaunches(repo)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [config, setConfig] = useState<FactoryConfig | null>(null)
  const loadConfig = useCallback(() => callFactory<FactoryConfig>("config.get", { repo }), [repo])
  useLoadWhile(true, loadConfig, setConfig)
  // Until the repo has a factory.yaml (or while that is unknown) no task starts.
  const missingConfig = config !== null && !config.exists
  const [showAllFinished, setShowAllFinished] = useState(false)
  const [showArchived, setShowArchived] = useState(false)
  const [archivedIds, updateArchived] = useStoredJson(`gigachad:factory-archived:${repo}`, parseArchived)
  const toggleArchived = (id: string) => updateArchived((ids) => (ids.includes(id) ? ids.filter((other) => other !== id) : [...ids, id]))
  const [deleteError, setDeleteError] = useState("")
  const deleteTask = async (id: string, force: boolean) => {
    await callFactory<FactoryTaskRemoval>("tasks.remove", force ? { id, force } : { id })
    await refresh()
  }
  const { groups, ungrouped } = useMemo(() => groupByTag(tasks), [tasks])

  return (
    // Lanes start at --surface, the level of the sidebar beside them
    <ElevationProvider base={1}>
      <div className="flex h-full flex-col">
        <InTopBar name="title">
          {/* Larger than a top-bar control, and set lower: the title of the page */}
          <h2 className="truncate pl-5 pt-3 text-xl font-semibold tracking-tight text-ink select-none">Factory <span className="font-normal text-ink-muted">· {baseName(repo)}</span></h2>
        </InTopBar>
        <InTopBar name="center">
          <div className="flex items-center gap-2">
            <button type="button" aria-label="Factory settings" onClick={() => setSettingsOpen(true)} className={ICON_BUTTON_CLASS}>
              <Settings className="h-4 w-4" />
            </button>
            <button type="button" disabled={missingConfig} onClick={() => setNewTaskOpen(true)} className={cn(PRIMARY_BUTTON_CLASS, "flex items-center gap-1.5")}>
              <Plus className="h-3.5 w-3.5" />
              New task
            </button>
          </div>
        </InTopBar>
        <ErrorText className="px-8 pt-3">{error ?? deleteError}</ErrorText>
        {missingConfig && config.draft !== null && (
          <ConfigDraft repo={repo} draft={config.draft} onSaved={(text) => setConfig({ exists: true, text, draft: null, source: [] })} />
        )}
        {!missingConfig && groups.length > 0 && (
          // Bounded and scrolling on its own, so open groups never squeeze the columns away
          <div className="max-h-[40vh] shrink-0 overflow-y-auto px-16 pt-6 scrollbar-thin">
            <TagTree
              groups={groups}
              renderTask={(task) => (
                <TaskCard
                  task={task}
                  archived={null}
                  onArchive={() => {}}
                  onOpen={() => onOpenTask(task.id)}
                  onDelete={(force) => deleteTask(task.id, force)}
                  onDeleteError={setDeleteError}
                />
              )}
            />
          </div>
        )}
        <div className={cn("min-h-0 flex-1 gap-16 overflow-x-auto px-16 pb-10 pt-6 scrollbar-thin", missingConfig ? "hidden" : "flex")}>
          {BOARD_COLUMNS.map((column) => {
            const inColumn = ungrouped.filter((task) => column.states.includes(shownState(task))).sort(boardOrder)
            const isFinished = column.id === "finished"
            const archivedCount = isFinished ? inColumn.filter((task) => archivedIds.includes(task.id)).length : 0
            // Finished shows what waits for inspection, or the archive while that is opened.
            const columnTasks = isFinished ? inColumn.filter((task) => archivedIds.includes(task.id) === (showArchived && archivedCount > 0)) : inColumn
            // Newest first, like the tasks: a refused start waits on the user, one on its way works.
            const columnLaunches = launches.filter((launch) => column.id === (launch.error === null ? "working" : "waiting")).reverse()
            const count = isFinished ? inColumn.length - archivedCount : columnTasks.length + columnLaunches.length
            const capped = isFinished && !showAllFinished && columnTasks.length > FINISHED_LIMIT
            const shown = capped ? columnTasks.slice(0, FINISHED_LIMIT) : columnTasks
            return (
              <BoardColumn
                key={column.id}
                role="region"
                aria-label={column.title}
                title={column.title}
                titleAs="h3"
                count={count}
                countExtra={archivedCount > 0 && (
                  <>
                    {count > 0 && <span aria-hidden className="text-[11px] text-ink-faint">|</span>}
                    <button
                      type="button"
                      aria-label="Archived"
                      aria-pressed={showArchived}
                      onClick={() => setShowArchived((open) => !open)}
                      className={cn(
                        "rounded text-[11px] tabular-nums hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted",
                        showArchived ? "text-ink" : "text-ink-faint",
                      )}
                    >
                      {archivedCount}
                    </button>
                  </>
                )}
                marker={<StatusDot tone={count === 0 ? "idle" : column.tone} />}
                className="min-w-[15rem] border-divider/30"
                bodyProps={{ className: "flex min-h-0 flex-1 flex-col gap-2.5" }}
                footer={<>
                  {capped && (
                    <button
                      type="button"
                      onClick={() => setShowAllFinished(true)}
                      className="rounded px-1 py-1 text-left text-[11px] text-ink-subtle hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
                    >
                      Show all {columnTasks.length}
                    </button>
                  )}
                </>}
              >
                {columnLaunches.map((launch) => (
                  <motion.div
                    key={launch.id}
                    layout
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.2 }}
                  >
                    <LaunchCard launch={launch} />
                  </motion.div>
                ))}
                {shown.map((task) => (
                  <motion.div
                    key={task.id}
                    layout
                    layoutId={task.id}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.2 }}
                  >
                    <TaskCard
                      task={task}
                      archived={isFinished ? archivedIds.includes(task.id) : null}
                      onArchive={() => toggleArchived(task.id)}
                      onOpen={() => onOpenTask(task.id)}
                      onDelete={(force) => deleteTask(task.id, force)}
                      onDeleteError={setDeleteError}
                    />
                  </motion.div>
                ))}
              </BoardColumn>
            )
          })}
        </div>
      </div>
      <NewTaskDialog repo={repo} open={newTaskOpen} onClose={() => setNewTaskOpen(false)} />
      <FactorySettingsDialog repo={repo} open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </ElevationProvider>
  )
}
