import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import type { FactoryTask } from "@/lib/domain/factory"
import type { PiCatalogProps } from "@/features/pi/PiTopBar"

const http = vi.hoisted(() => ({ post: vi.fn<(path: string, body: unknown) => Promise<unknown>>() }))

vi.mock("@/lib/backend/api/http", () => http)
vi.mock("@/features/pi/PiTab", () => ({
  PiSessionView: ({ sessionId, cwd, files }: { sessionId: string; cwd: string; files: { cwd: string } }) => <div data-testid="session" data-session={sessionId} data-cwd={cwd} data-files={files.cwd} />,
}))
vi.mock("@/features/pi/PiTopBar", () => ({ PiPanelBar: () => null }))

import { TaskPiChat } from "@/features/factory/TaskPiChat"

const TASK: FactoryTask = {
  id: "t1", repo: "/home/u/git/demo", session_id: "s1", session_path: "/sessions/s1.jsonl", worktree: "/home/u/git/worktrees/demo/t1", branch: "factory/t1",
  title: "T", created_at: "", factory: { state: "blocked", round: 1, reason: "", at: "" }, live: { running: false, waiting: false, attached: 0 },
}

const chat = (task: FactoryTask) => <TaskPiChat task={task} bar={null} modelProps={{} as PiCatalogProps} />

beforeEach(() => {
  http.post.mockReset().mockResolvedValue(TASK)
})
afterEach(cleanup)

describe("TaskPiChat", () => {
  it("shows the task's own live session in its worktree without opening anything", () => {
    render(chat(TASK))

    const session = screen.getByTestId("session")
    expect(session.dataset).toMatchObject({ session: "s1", cwd: "/home/u/git/worktrees/demo/t1", files: "/home/u/git/worktrees/demo/t1" })
    expect(http.post).not.toHaveBeenCalled()
  })

  it("runs a current-checkout task's session in the repo", () => {
    render(chat({ ...TASK, worktree: null, branch: null }))

    expect(screen.getByTestId("session").dataset.cwd).toBe("/home/u/git/demo")
  })

  it("opens a stopped session once before showing it", async () => {
    const stopped = { ...TASK, live: null }
    const rendered = render(chat(stopped))
    expect(screen.queryByTestId("session")).toBeNull()
    expect(screen.getByRole("status").textContent).toBe("Loading…")

    await waitFor(() => expect(screen.getByTestId("session")).toBeTruthy())
    expect(http.post).toHaveBeenCalledWith("/factory/tasks/t1/session", {})

    // The view stays when the child stops later
    rendered.rerender(chat(stopped))
    expect(screen.getByTestId("session")).toBeTruthy()
    expect(http.post).toHaveBeenCalledTimes(1)
  })

  it("shows why a stopped session could not be opened, and the session once its child runs", async () => {
    http.post.mockRejectedValue(new Error("pi-host is down"))
    const rendered = render(chat({ ...TASK, live: null }))
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("pi-host is down"))

    rendered.rerender(chat(TASK))
    expect(screen.getByTestId("session")).toBeTruthy()
  })
})
