import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { TabContext, type Tab } from "@/contexts/TabContext"
import type { PiRepository, PiSearchHit, PiSessionInfo } from "@/lib/backend/api"
import type { PiSessionStatus } from "../sessionStore"
import { groupPiSessions, PiSessionPicker, RECENT_SESSION_MS } from "../PiSessionPicker"

const pi = vi.hoisted(() => ({
  sessions: [] as PiSessionInfo[],
  repositories: [] as PiRepository[],
  statuses: {} as Record<string, PiSessionStatus>,
  search: vi.fn<(query: string) => Promise<PiSearchHit[]>>(),
  archive: vi.fn<(sessionId: string) => Promise<PiSessionInfo>>(),
}))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useHostAccess: () => true,
}))
vi.mock("../sessionStore", () => ({
  usePiSessions: () => ({ sessions: pi.sessions, loading: false, error: null }),
  usePiRepositories: () => pi.repositories,
  usePiSessionStatus: (sessionId: string): PiSessionStatus => pi.statuses[sessionId] ?? { running: false, waiting: false, unread: false },
  refreshPiSessions: vi.fn(),
  searchPiSessions: (query: string) => pi.search(query),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  archivePiSession: (sessionId: string) => pi.archive(sessionId),
}))

const HOUR = 60 * 60 * 1000

function piSession(sessionId: string, ageMs: number, extra: Partial<PiSessionInfo> = {}): PiSessionInfo {
  const at = new Date(Date.now() - ageMs).toISOString()
  return {
    sessionId, sessionPath: `/s/${sessionId}.jsonl`, cwd: "/git/proj", machine: "local", createdAt: at, modifiedAt: at,
    firstMessage: sessionId, messageCount: 1, live: null, pinned: false, archived: false, ...extra,
  }
}

const live = { running: true, waiting: false, attached: 1 }

const TAB: Tab = { id: "tab", name: null, chatId: "chat", historyFile: null, title: null, config: {} as Tab["config"], appMode: "chat", projectSlug: null, workspaceItem: null }

function renderPicker(onOpenItem = vi.fn()) {
  render(<TabContext.Provider value={{ tab: TAB, isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
    <PiSessionPicker onOpenItem={onOpenItem} />
  </TabContext.Provider>)
  fireEvent.keyDown(window, { key: "a", altKey: true })
  return onOpenItem
}

afterEach(() => {
  cleanup()
  pi.sessions = []
  pi.repositories = []
  pi.statuses = {}
  pi.search.mockReset()
  pi.archive.mockReset()
})

describe("groupPiSessions", () => {
  it("puts running and waiting sessions first, then the ones of the last day, then the rest, each newest first", () => {
    const now = Date.now()
    const groups = groupPiSessions([
      piSession("old", RECENT_SESSION_MS + HOUR),
      piSession("fresh", HOUR),
      piSession("busy", 3 * RECENT_SESSION_MS, { live }),
      piSession("asking", 2 * HOUR, { live: { running: false, waiting: true, attached: 1 } }),
      piSession("fresher", 1000),
    ], now)
    expect(groups.running.map((session) => session.sessionId)).toEqual(["asking", "busy"])
    expect(groups.recent.map((session) => session.sessionId)).toEqual(["fresher", "fresh"])
    expect(groups.older.map((session) => session.sessionId)).toEqual(["old"])
  })
})

describe("PiSessionPicker", () => {
  it("opens on Alt+A with Running, Recent and older sessions behind a row, marks them, and opens a picked one", () => {
    pi.repositories = [{ name: "proj", path: "/git/proj" }]
    pi.sessions = [
      piSession("busy", HOUR, { live }),
      piSession("fresh", HOUR, { cwd: "/git/worktrees/proj/feat" }),
      piSession("old", RECENT_SESSION_MS + HOUR),
    ]
    pi.statuses = { busy: { running: true, waiting: false, unread: false }, fresh: { running: false, waiting: false, unread: true } }
    const onOpenItem = renderPicker()

    expect(within(screen.getByRole("group", { name: "Running" })).getByRole("option", { name: /busy/ })).toBeInTheDocument()
    expect(within(screen.getByRole("group", { name: "Recent" })).getByRole("option", { name: /fresh/ })).toBeInTheDocument()
    expect(within(screen.getByRole("option", { name: /busy/ })).getByRole("img", { name: "Running" })).toBeInTheDocument()
    expect(within(screen.getByRole("option", { name: /fresh/ })).getByRole("img", { name: "Unread" })).toBeInTheDocument()
    expect(screen.getByRole("option", { name: /fresh/ })).toHaveTextContent("feat")

    fireEvent.click(screen.getByRole("option", { name: "Show 1 older" }))
    expect(screen.getByRole("option", { name: /old/ })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("option", { name: /fresh/ }))
    expect(onOpenItem).toHaveBeenCalledWith({ kind: "pi", path: "proj", projectSlug: null, pi: { sessionId: "fresh", repo: "proj" } })
  })

  it("searches the transcripts while typing and shows the matched text of each hit beside title matches", async () => {
    pi.repositories = [{ name: "proj", path: "/git/proj" }]
    const titled = piSession("refactor tabs", HOUR)
    const other = piSession("unrelated", 2 * HOUR)
    pi.sessions = [titled, other]
    pi.search.mockResolvedValue([
      { session: titled, snippet: "about refactor", entryId: "e1" },
      { session: other, snippet: "…we should refactor the store…", entryId: "e2" },
    ])
    const onOpenItem = renderPicker()

    fireEvent.change(screen.getByRole("combobox", { name: "Search sessions…" }), { target: { value: "refactor" } })
    await waitFor(() => expect(pi.search).toHaveBeenCalledWith("refactor"))
    const content = await screen.findByRole("group", { name: "Content" })
    expect(within(screen.getByRole("group", { name: "Sessions" })).getByRole("option", { name: /refactor tabs/ })).toBeInTheDocument()
    const hit = within(content).getByRole("option")
    expect(hit).toHaveTextContent("unrelated")
    expect(hit).toHaveTextContent("…we should refactor the store…")

    fireEvent.click(hit)
    expect(onOpenItem).toHaveBeenCalledWith({ kind: "pi", path: "proj", projectSlug: null, pi: { sessionId: "unrelated", repo: "proj" } })
  })

  it("archives the highlighted session on Delete at the search's end", async () => {
    pi.sessions = [piSession("first", HOUR), piSession("second", 2 * HOUR)]
    pi.archive.mockImplementation(async (sessionId) => piSession(sessionId, HOUR, { archived: true }))
    renderPicker()

    await act(async () => { fireEvent.keyDown(screen.getByRole("combobox", { name: "Search sessions…" }), { key: "Delete" }) })
    expect(pi.archive).toHaveBeenCalledWith("first")
    expect(screen.queryByRole("option", { name: /first/ })).not.toBeInTheDocument()
    expect(screen.getByRole("option", { name: /second/ })).toBeInTheDocument()
  })
})
