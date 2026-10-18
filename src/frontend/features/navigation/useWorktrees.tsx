"use client"

import { useCallback, useMemo, useState, useSyncExternalStore } from "react"
import type { FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { InlineConfirm } from "@/components/ui/InlineConfirm"
import { ApiError, createWorktree, deleteWorktreeBranch, fetchWorktrees, removeWorktree, type Worktree, type WorktreeList } from "@/lib/backend/api"
import { baseName } from "@/lib/paths"

const ACTIVE_STORAGE_KEY = "active_worktrees"
const ACTIVE_EVENT = "active-worktrees"

function subscribeActive(callback: () => void) {
  window.addEventListener(ACTIVE_EVENT, callback)
  return () => window.removeEventListener(ACTIVE_EVENT, callback)
}

function parseActive(stored: string): Record<string, string> {
  try { return JSON.parse(stored) as Record<string, string> } catch { return {} }
}

/** Questions after the trash click: force a dirty worktree, offer deleting its branch, force an unmerged branch. */
type RemovalStep = "force" | "branch" | "unmerged"

interface Removal {
  mount: string
  worktree: Worktree
  /** null while git works. */
  step: RemovalStep | null
  error: string
}

const QUESTIONS: Record<RemovalStep, (branch: string) => string> = {
  force: () => "Uncommitted changes are lost. Force remove?",
  branch: (branch) => `Delete branch ${branch}?`,
  unmerged: () => "Branch has unmerged commits. Delete anyway?",
}

const errorText = (error: unknown) => error instanceof Error ? error.message : String(error)
const isConflict = (error: unknown) => error instanceof ApiError && error.status === 409

/** `3 changed ↑2 ↓1`: uncommitted files, then commits ahead of and behind the base branch. */
export function worktreeStatus(worktree: Worktree): string | undefined {
  const parts = [
    worktree.changed_files ? `${worktree.changed_files} changed` : "",
    worktree.ahead ? `↑${worktree.ahead}` : "",
    worktree.behind ? `↓${worktree.behind}` : "",
  ].filter(Boolean)
  return parts.length ? parts.join(" ") : undefined
}

/** Worktrees of every clone mount: which one each mount shows, and its sidebar "worktrees" folder. */
export function useWorktrees(mounts: string[]) {
  const [lists, setLists] = useState<Record<string, WorktreeList>>({})
  const [removal, setRemoval] = useState<Removal | null>(null)
  const stored = useSyncExternalStore(subscribeActive, () => localStorage.getItem(ACTIVE_STORAGE_KEY) ?? "{}", () => "{}")
  const active = useMemo(() => parseActive(stored), [stored])
  const mountsKey = mounts.join("\n")

  const load = useCallback((mount: string) => fetchWorktrees(mount).then(
    (list) => setLists((current) => ({ ...current, [mount]: list })),
    () => {},
  ), [])
  const reload = useCallback(() => { for (const mount of mountsKey.split("\n").filter(Boolean)) void load(mount) }, [mountsKey, load])

  const setActive = (mount: string, path: string | null) => {
    const next = { ...parseActive(localStorage.getItem(ACTIVE_STORAGE_KEY) ?? "{}") }
    if (path) next[mount] = path
    else delete next[mount]
    localStorage.setItem(ACTIVE_STORAGE_KEY, JSON.stringify(next))
    window.dispatchEvent(new Event(ACTIVE_EVENT))
  }

  /** The stored worktree while it still exists, else the clone. */
  const shown = (mount: string): Worktree | undefined => {
    const worktrees = lists[mount]?.worktrees ?? []
    return worktrees.find((worktree) => worktree.path === active[mount]) ?? worktrees.find((worktree) => worktree.is_main)
  }

  const finish = (mount: string) => {
    setRemoval(null)
    void load(mount)
  }

  const removed = (current: Removal) => {
    if (active[current.mount] === current.worktree.path) setActive(current.mount, null)
    if (current.worktree.branch) setRemoval({ ...current, step: "branch", error: "" })
    else finish(current.mount)
  }

  const startRemoval = async (mount: string, worktree: Worktree) => {
    const current: Removal = { mount, worktree, step: null, error: "" }
    setRemoval(current)
    try {
      await removeWorktree(mount, worktree.path)
      removed(current)
    } catch (error) {
      setRemoval(isConflict(error) ? { ...current, step: "force" } : { ...current, error: errorText(error) })
    }
  }

  const answer = async (current: Removal, yes: boolean) => {
    if (!yes || !current.step) return finish(current.mount)
    setRemoval({ ...current, step: null, error: "" })
    try {
      if (current.step === "force") {
        await removeWorktree(current.mount, current.worktree.path, true)
        return removed(current)
      }
      await deleteWorktreeBranch(current.mount, current.worktree.branch ?? "", current.step === "unmerged")
      finish(current.mount)
    } catch (error) {
      setRemoval(current.step === "branch" && isConflict(error) ? { ...current, step: "unmerged" } : { ...current, error: errorText(error) })
    }
  }

  const removalDetail = (current: Removal) => <>
    {current.step && <InlineConfirm
      question={QUESTIONS[current.step](current.worktree.branch ?? "")}
      onYes={() => void answer(current, true)} onNo={() => void answer(current, false)}
    />}
    {!current.step && !current.error && <p className="py-0.5 text-[11px] text-ink-faint">Removing…</p>}
    {current.error && <p role="alert" className="flex items-center gap-1 py-0.5 text-[11px] text-danger">
      <span className="min-w-0 flex-1">{current.error}</span>
      <button onClick={() => finish(current.mount)} className="rounded px-1 text-ink-muted hover:bg-hover hover:text-ink">Dismiss</button>
    </p>}
  </>

  const folder = (mount: string): FileTreeItem<string> | null => {
    const list = lists[mount]
    if (!list?.is_clone) return null
    const current = shown(mount)
    const base = list.default_base
    return {
      id: `worktrees:${mount}`, label: "worktrees", type: "folder",
      create: base ? {
        label: `New worktree of ${baseName(mount)}`, placeholder: "Worktree name",
        choice: { label: "from", options: list.branches, initial: base },
        onSubmit: async (name, from) => {
          const created = await createWorktree(mount, name, from ?? base)
          setActive(mount, created.path)
          await load(mount)
        },
      } : undefined,
      children: list.worktrees.map((worktree) => ({
        id: `worktree:${worktree.path}`, label: worktree.branch ?? baseName(worktree.path), type: "element",
        badge: worktreeStatus(worktree),
        isActive: worktree.path === current?.path,
        onSelect: () => setActive(mount, worktree.is_main ? null : worktree.path),
        onDelete: worktree.is_main || removal ? undefined : () => void startRemoval(mount, worktree),
        detail: removal?.worktree.path === worktree.path ? removalDetail(removal) : undefined,
      })),
    }
  }

  /** Worktree paths the mounts show instead of their clone, for the file tree request. */
  const activeWorktrees = mountsKey.split("\n").map((mount) => active[mount]).filter(Boolean)

  /** Branch of the worktree a mount shows, unless it shows the clone. */
  const shownBranch = (mount: string): string | undefined => {
    const worktree = shown(mount)
    return worktree && !worktree.is_main ? worktree.branch ?? baseName(worktree.path) : undefined
  }

  return { activeWorktrees, folder, shownBranch, reload }
}
