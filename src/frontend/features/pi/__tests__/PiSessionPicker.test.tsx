import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { TabContext, type Tab } from "@/contexts/TabContext"
import { ApiError, type PiRepository, type PiSearchHit, type PiSessionInfo } from "@/lib/backend/api"
import type { PiSessionStatus } from "../sessionStore"
import { PiSessionPicker } from "../PiSessionPicker"

const pi = vi.hoisted(() => ({
  sessions: [] as PiSessionInfo[],
  repositories: [] as PiRepository[],
  statuses: {} as Record<string, PiSessionStatus>,
  search: vi.fn<(query: string, filter?: { profile?: string }) => Promise<PiSearchHit[]>>(),
  listFilters: [] as unknown[],
  archive: vi.fn<(sessionId: string) => Promise<PiSessionInfo>>(),
  focus: vi.fn(),
}))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useHostAccess: () => true,
}))
vi.mock("../sessionStore", () => ({
  usePiSessions: (filter: unknown) => {
    pi.listFilters.push(filter)
    return { sessions: pi.sessions, loading: false, error: null }
  },
  usePiRepositories: () => pi.repositories,
  usePiSessionStatus: (sessionId: string): PiSessionStatus => pi.statuses[sessionId] ?? { running: false, waiting: false, unread: false },
  refreshPiSessions: vi.fn(),
  searchPiSessions: (query: string, filter?: { profile?: string }) => pi.search(query, filter),
}))
vi.mock("../searchFocus", () => ({ focusPiSearchHit: (...args: unknown[]) => pi.focus(...args) }))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  archivePiSession: (sessionId: string) => pi.archive(sessionId),
}))

const HOUR = 60 * 60 * 1000

function piSession(sessionId: string, ageMs: number, extra: Partial<PiSessionInfo> = {}): PiSessionInfo {
  const at = new Date(Date.now() - ageMs).toISOString()
  return {
    sessionId, sessionPath: `/s/${sessionId}.jsonl`, cwd: "/git/proj", machine: "local", createdAt: at, modifiedAt: at,
    firstMessage: sessionId, messageCount: 1, tokens: 0, live: null, pinned: false, archived: false, ...extra,
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
  pi.listFilters = []
  pi.search.mockReset()
  pi.archive.mockReset()
})

describe("PiSessionPicker", () => {
  it("opens on Alt+A with every session newest first, its age, tokens and directory, marks them, and opens a picked one", () => {
    pi.repositories = [{ name: "proj", path: "/git/proj" }]
    pi.sessions = [
      piSession("old", 3 * HOUR),
      piSession("busy", 2 * HOUR, { live, tokens: 84_300 }),
      piSession("fresh", HOUR, { cwd: "/git/worktrees/proj/feat" }),
    ]
    pi.statuses = { busy: { running: true, waiting: false, unread: false }, fresh: { running: false, waiting: false, unread: true } }
    const onOpenItem = renderPicker()

    expect(screen.getAllByRole("option").map((option) => option.textContent?.split(/\dh/)[0])).toEqual(["fresh", "busy", "old"])
    expect(screen.getByRole("option", { name: /busy/ })).toHaveTextContent("2h84k/git/proj")
    expect(within(screen.getByRole("option", { name: /busy/ })).getByRole("img", { name: "Running" })).toBeInTheDocument()
    expect(within(screen.getByRole("option", { name: /fresh/ })).getByRole("img", { name: "Unread" })).toBeInTheDocument()
    expect(screen.getByRole("option", { name: /fresh/ })).toHaveTextContent("feat")

    fireEvent.click(screen.getByRole("option", { name: /fresh/ }))
    expect(onOpenItem).toHaveBeenCalledWith({ kind: "pi", path: "proj", projectSlug: null, pi: { sessionId: "fresh", repo: "proj" } })
  })

  it("searches the transcripts while typing and shows the matched text of each hit beside title matches", async () => {
    pi.repositories = [{ name: "proj", path: "/git/proj" }]
    const titled = piSession("refactor tabs", HOUR)
    const other = piSession("unrelated", 2 * HOUR)
    pi.sessions = [titled, other]
    pi.search.mockResolvedValue([
      { session: titled, snippet: "about refactor", entryId: "e1", matches: [] },
      { session: other, snippet: "…we should refactor the store…", entryId: "e2", matches: [] },
    ])
    const onOpenItem = renderPicker()

    fireEvent.change(screen.getByRole("combobox", { name: "Search sessions…" }), { target: { value: "refactor" } })
    await waitFor(() => expect(pi.search).toHaveBeenCalledWith("refactor", { profile: "coding" }))
    const content = await screen.findByRole("group", { name: "Content" })
    expect(within(screen.getByRole("group", { name: "Sessions" })).getByRole("option", { name: /refactor tabs/ })).toBeInTheDocument()
    const hit = within(content).getByRole("option")
    expect(hit).toHaveTextContent("unrelated")
    expect(hit).toHaveTextContent("…we should refactor the store…")

    fireEvent.click(hit)
    expect(onOpenItem).toHaveBeenCalledWith({ kind: "pi", path: "proj", projectSlug: null, pi: { sessionId: "unrelated", repo: "proj" } })
  })

  it("lists a session's hits by content and opens the picked one at that hit", async () => {
    pi.repositories = [{ name: "proj", path: "/git/proj" }]
    const other = piSession("unrelated", 2 * HOUR)
    pi.sessions = [other]
    const first = { entryId: "e2", ordinal: 0, quality: 1, before: "…we should ", match: "refactor", after: " the store…" }
    const second = { entryId: "e5", ordinal: 1, quality: 0.8, before: "and ", match: "refactoring", after: " later" }
    pi.search.mockResolvedValue([{ session: other, snippet: "…we should refactor the store…", entryId: "e2", matches: [first, second] }])
    const onOpenItem = renderPicker()

    fireEvent.change(screen.getByRole("combobox", { name: "Search sessions…" }), { target: { value: " refactor " } })
    const rows = within(await screen.findByRole("group", { name: "Content" })).getAllByRole("option")
    expect(rows).toHaveLength(2)
    expect(rows[0]).toHaveTextContent("unrelated")
    expect(rows[1]).toHaveTextContent("and refactoring later")
    expect(rows[1]).not.toHaveTextContent("unrelated")

    fireEvent.click(rows[1])
    expect(pi.focus).toHaveBeenCalledWith("unrelated", "refactor", second, [first, second])
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

  it("says why an archive failed in words, not pi-host's code", async () => {
    pi.sessions = [piSession("first", HOUR)]
    pi.archive.mockRejectedValue(new ApiError("live", 409))
    renderPicker()

    await act(async () => { fireEvent.keyDown(screen.getByRole("combobox", { name: "Search sessions…" }), { key: "Delete" }) })
    expect(screen.getByText("Session is running")).toBeInTheDocument()
    expect(screen.queryByText("live")).not.toBeInTheDocument()
    expect(screen.getByRole("option", { name: /first/ })).toBeInTheDocument()
  })

  it("lists and searches only coding sessions (G5)", async () => {
    pi.search.mockResolvedValue([])
    renderPicker()
    expect(pi.listFilters.length).toBeGreaterThan(0)
    for (const filter of pi.listFilters) expect(filter).toEqual({ profile: "coding" })

    fireEvent.change(screen.getByRole("combobox", { name: "Search sessions…" }), { target: { value: "notes" } })
    await waitFor(() => expect(pi.search).toHaveBeenCalledWith("notes", { profile: "coding" }))
  })
})
