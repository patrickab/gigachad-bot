import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import type { PiRepository, PiSessionInfo } from "@/lib/backend/api"
import { piStartDirectories, PiWorkspacePicker } from "../PiWorkspacePicker"

const pi = vi.hoisted(() => ({
  repositories: [] as PiRepository[],
  hostAccess: true,
}))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useHostAccess: () => pi.hostAccess,
}))
vi.mock("../sessionStore", () => ({
  usePiSessions: () => ({ sessions: [], loading: false, error: null }),
  usePiRepositories: () => pi.repositories,
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fileRootTree: async () => ({ enabled: true, tree: [] }),
}))

const HOUR = 60 * 60 * 1000

function piSession(sessionId: string, cwd: string, ageMs: number, firstMessage = sessionId): PiSessionInfo {
  const at = new Date(Date.now() - ageMs).toISOString()
  return { sessionId, sessionPath: `/s/${sessionId}.jsonl`, cwd, machine: "local", createdAt: at, modifiedAt: at, firstMessage, messageCount: 1, tokens: 0, live: null, pinned: false, archived: false }
}

const REPOSITORIES: PiRepository[] = [
  { name: "alpha", path: "/home/u/git/alpha" },
  { name: "beta", path: "/home/u/git/beta" },
  { name: "gamma", path: "/home/u/git/gamma" },
]

afterEach(() => {
  cleanup()
  pi.repositories = []
  pi.hostAccess = true
})

/** The picker opened by its trigger, once its file mounts are read. */
async function openPicker(folder: string | null = null, onChange = vi.fn(), onOpenSession = vi.fn()) {
  render(<PiWorkspacePicker folder={folder} onChange={onChange} onOpenSession={onOpenSession} />)
  await act(async () => fireEvent.click(screen.getByRole("button", { name: `Project: ${folder === null ? "none" : folder.split("/").at(-1)}` })))
  return onChange
}

describe("piStartDirectories", () => {
  it("orders checkouts and mounts by their newest session, worktrees counting for their checkout, the unused ones by name", () => {
    const sessions = [
      piSession("old", "/home/u/git/gamma", 5 * HOUR),
      piSession("new", "/home/u/git/worktrees/beta/feature", HOUR),
      piSession("mounted", "/mnt/notes/sub", 3 * HOUR),
      piSession("older", "/home/u/git/gamma/src", 9 * HOUR),
    ]
    const directories = piStartDirectories(REPOSITORIES, ["/mnt/notes/", "/home/u/git/alpha"], sessions)
    expect(directories.map((directory) => [directory.label, directory.latest?.sessionId ?? null])).toEqual([
      ["beta", "new"], ["notes", "mounted"], ["gamma", "old"], ["alpha", null],
    ])
    expect(directories.map((directory) => directory.path)).toEqual(["/home/u/git/beta", "/mnt/notes", "/home/u/git/gamma", "/home/u/git/alpha"])
  })
})

describe("PiWorkspacePicker", () => {
  it("hands the picked checkout's path to the caller, None clears it", async () => {
    pi.repositories = REPOSITORIES
    const onChange = await openPicker()
    fireEvent.click(screen.getByRole("option", { name: /Not yet used/ }))
    fireEvent.click(screen.getByRole("option", { name: "beta" }))
    expect(onChange).toHaveBeenLastCalledWith("/home/u/git/beta")

    cleanup()
    const cleared = await openPicker("/home/u/git/beta")
    fireEvent.click(screen.getByRole("option", { name: "None" }))
    expect(cleared).toHaveBeenLastCalledWith(null)
  })

  it("is hidden without host access", () => {
    pi.hostAccess = false
    render(<PiWorkspacePicker folder={null} onChange={vi.fn()} onOpenSession={vi.fn()} />)
    expect(screen.queryByRole("button", { name: "Project: none" })).toBeNull()
  })
})
