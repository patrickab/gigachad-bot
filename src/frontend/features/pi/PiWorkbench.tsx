"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { FolderGit2, Plus } from "lucide-react"
import { FilePanelProvider, FilePanelSlot, FilePanelToggle } from "@/features/editor/FilePanel"
import { SidePanel, SidePanelScroll } from "@/components/ui/SidePanel"
import { StatusDot, type StatusTone } from "@/components/ui/StatusDot"
import { fetchPiRepositories, fetchPiSessions, openPiSession, type PiRepository, type PiResume, type PiSessionInfo } from "@/lib/backend/api"
import { fuzzyScore } from "@/lib/fuzzy"
import { cn } from "@/lib/utils"
import { NEW_SESSION_OPTIONS, piOpenError, PiSessionView } from "./PiTab"
import { PiSessionControlsBar } from "./PiTopBar"

const REPO_PARAM = "repo"
const SESSION_PARAM = "session"
const SIDEBAR_WIDTH = 240
/** How often the open repo's session list refreshes, for the live dots. */
const SESSIONS_REFRESH_MS = 5_000

const ROW_CLASS = "flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-sm text-ink-muted hover:bg-ink/5 hover:text-ink focus-visible:outline focus-visible:outline-1"

/** Dot for a session's live child: pulsing while it runs, solid while it waits on a dialog, none when no child runs. */
function liveTone(session: PiSessionInfo): StatusTone | null {
  if (!session.live) return null
  return session.live.waiting ? "attention" : session.live.running ? "active" : "ready"
}

function setUrlParams(params: Record<string, string | null>) {
  const url = new URL(window.location.href)
  for (const [name, value] of Object.entries(params)) {
    if (value === null) url.searchParams.delete(name)
    else url.searchParams.set(name, value)
  }
  window.history.replaceState(null, "", url)
}

function SessionRows({ sessions, activeId, onOpen }: { sessions: PiSessionInfo[]; activeId: string | null; onOpen: (resume: PiResume) => void }) {
  return (
    <div className="mb-1 ml-3 flex flex-col">
      <button type="button" onClick={() => onOpen(null)} className={ROW_CLASS}>
        <Plus size={12} aria-hidden="true" className="shrink-0" />
        <span className="truncate">New session</span>
      </button>
      {sessions.map((session) => {
        const tone = liveTone(session)
        return (
          <button
            key={session.sessionId}
            type="button"
            aria-current={session.sessionId === activeId ? "page" : undefined}
            onClick={() => onOpen({ sessionPath: session.sessionPath })}
            className={cn(ROW_CLASS, session.sessionId === activeId && "bg-ink/8 text-ink")}
          >
            <span className="flex w-3 shrink-0 justify-center">{tone && <StatusDot tone={tone} />}</span>
            <span className="truncate">{session.firstMessage || "New session"}</span>
          </button>
        )
      })}
    </div>
  )
}

function RepositorySidebar({ repositories, selected, sessions, activeId, onSelect, onOpen }: {
  repositories: PiRepository[]
  selected: string | null
  sessions: PiSessionInfo[]
  activeId: string | null
  onSelect: (name: string) => void
  onOpen: (resume: PiResume) => void
}) {
  const [query, setQuery] = useState("")
  const shown = useMemo(() => {
    if (!query) return repositories
    return repositories
      .map((repository) => ({ repository, score: fuzzyScore(repository.name, query.toLowerCase()) }))
      .filter((entry): entry is { repository: PiRepository; score: number } => entry.score !== null)
      .sort((left, right) => right.score - left.score)
      .map((entry) => entry.repository)
  }, [repositories, query])

  return (
    <SidePanel side="left" open width={SIDEBAR_WIDTH} label="Repositories">
      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Filter"
        aria-label="Filter repositories"
        className="m-2 rounded-lg bg-transparent px-2.5 py-1.5 text-sm text-ink outline-none placeholder:text-ink-faint focus-visible:ring-1 focus-visible:ring-divider"
      />
      <SidePanelScroll className="px-1.5 pb-2">
        <nav aria-label="Repositories">
          {shown.map((repository) => (
            <div key={repository.name}>
              <button
                type="button"
                title={repository.path}
                aria-expanded={repository.name === selected}
                onClick={() => onSelect(repository.name)}
                className={cn(ROW_CLASS, repository.name === selected && "text-ink")}
              >
                <FolderGit2 size={14} aria-hidden="true" className="shrink-0" />
                <span className="truncate">{repository.name}</span>
              </button>
              {repository.name === selected && <SessionRows sessions={sessions} activeId={activeId} onOpen={onOpen} />}
            </div>
          ))}
        </nav>
      </SidePanelScroll>
    </SidePanel>
  )
}

/**
 * Prototype: Pi agent sessions in any repository under `~/git`, served by pi-host and rendered with gigachad's chat UI.
 *
 * - `?repo=` picks the repository, `?session=` the open session (a reload resumes it), file links open in the right file panel (Neovim)
 * - Picking a repo continues its newest session, the sidebar lists its sessions with New, click-to-resume and live dots
 */
export function PiWorkbench() {
  const [repositories, setRepositories] = useState<PiRepository[]>([])
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(() => new URLSearchParams(window.location.search).get(REPO_PARAM))
  const [sessions, setSessions] = useState<PiSessionInfo[]>([])
  const [active, setActive] = useState<{ sessionId: string; opening: number } | null>(null)

  useEffect(() => {
    fetchPiRepositories().then(setRepositories, (reason: unknown) => setError(String(reason)))
  }, [])

  const open = useCallback(async (repo: string, resume: PiResume) => {
    setError(null)
    try {
      const session = await openPiSession(repo, resume, resume === null ? NEW_SESSION_OPTIONS : {})
      setUrlParams({ [SESSION_PARAM]: session.sessionId })
      // A fresh `opening` remounts the view, so reopening an exited session attaches again
      setActive((current) => ({ sessionId: session.sessionId, opening: (current?.opening ?? 0) + 1 }))
      setSessions(await fetchPiSessions(repo))
    } catch (reason) {
      setError(piOpenError(reason))
    }
  }, [])

  // On a repo pick: resume the session in the URL if it belongs to the repo, else continue the newest. Then keep its list fresh
  useEffect(() => {
    if (!selected) return
    let cancelled = false
    const wanted = new URLSearchParams(window.location.search).get(SESSION_PARAM)
    fetchPiSessions(selected).then((listed) => {
      if (cancelled) return
      setSessions(listed)
      const session = listed.find((candidate) => candidate.sessionId === wanted)
      void open(selected, session ? { sessionPath: session.sessionPath } : "continue")
    }, (reason: unknown) => setError(String(reason)))
    const timer = setInterval(() => {
      fetchPiSessions(selected).then((listed) => cancelled || setSessions(listed), () => undefined)
    }, SESSIONS_REFRESH_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [selected, open])

  const select = (name: string) => {
    if (name === selected) return
    setUrlParams({ [REPO_PARAM]: name, [SESSION_PARAM]: null })
    setActive(null)
    setSessions([])
    setSelected(name)
  }

  const repository = repositories.find((candidate) => candidate.name === selected)

  return (
    <main className="flex h-dvh bg-surface text-ink">
      <RepositorySidebar
        repositories={repositories}
        selected={selected}
        sessions={sessions}
        activeId={active?.sessionId ?? null}
        onSelect={select}
        onOpen={(resume) => selected && void open(selected, resume)}
      />
      <FilePanelProvider>
        <div className="flex min-w-0 flex-1">
          {repository && active && !error ? (
            <PiSessionView
              key={`${active.sessionId}:${active.opening}`}
              sessionId={active.sessionId}
              cwd={repository.path}
              files={{ repo: repository.name }}
              onBranch={(forkOf) => open(repository.name, { forkOf })}
              renderHeader={({ tone, sessionName, controls }) => (
                <header className="flex h-11 shrink-0 items-center gap-2 border-b border-divider/50 px-4 text-sm">
                  <StatusDot tone={tone} />
                  <span className="font-medium text-ink">{repository.name}</span>
                  {sessionName && <span className="truncate text-ink-muted">{sessionName}</span>}
                  <div className="ml-auto"><PiSessionControlsBar controls={controls} /></div>
                  <FilePanelToggle />
                </header>
              )}
            />
          ) : (
            <p role={error ? "alert" : undefined} className={cn("m-auto text-sm", error ? "text-danger" : "text-ink-faint")}>
              {error ?? (selected ? "" : "Pick a repository")}
            </p>
          )}
          <FilePanelSlot />
        </div>
      </FilePanelProvider>
    </main>
  )
}
