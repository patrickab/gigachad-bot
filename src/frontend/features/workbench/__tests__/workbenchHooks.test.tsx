import { StrictMode, useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render, renderHook, screen, waitFor } from "@testing-library/react"

const api = vi.hoisted(() => ({
  loadFileViewerText: vi.fn<(path: string) => Promise<string>>(),
  prepareWorkbench: vi.fn<(path: string) => Promise<{ path: string; cwd: string }>>(),
}))
vi.mock("@/lib/backend/api", () => api)
const follow = vi.hoisted(() => ({ paths: [] as readonly string[], changed: null as ((path: string) => void) | null }))
vi.mock("@/hooks/useFollowFiles", () => ({
  useFollowFiles: (files: readonly string[], onChange: (path: string) => void) => {
    follow.paths = files
    follow.changed = onChange
  },
}))

import { useFollowBuffers } from "../useFollowBuffers"
import { usePreparedWorkbench } from "../usePreparedWorkbench"

const PATH = "/data/workbench/markdown/proj/plan/plan.md"
const CWD = "/data/workbench/markdown"

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

/** A tab's workbench item as `WorkspaceSurface` holds it: the moved path replaces the item path, the workbench mounts once prepared. */
function PreparedItem({ initialPath, onMoved }: { initialPath: string; onMoved: (path: string) => void }) {
  const [path, setPath] = useState(initialPath)
  const { prepared, error } = usePreparedWorkbench(path, (moved) => {
    setPath(moved)
    onMoved(moved)
  })
  return <p>{error ?? (prepared ? `mounted ${prepared.path}` : "preparing")}</p>
}

/** A prepare answer the test hands out when it wants. */
function deferred() {
  let resolve: (value: { path: string; cwd: string }) => void = () => {}
  const promise = new Promise<{ path: string; cwd: string }>((done) => { resolve = done })
  return { promise, resolve }
}

// Prepared paths stay known for the module's lifetime, so every test uses its own paths.
describe("usePreparedWorkbench", () => {
  it("reports a moved document and treats its new path as prepared", async () => {
    const disk = "/data/workbench/markdown/moved/plan/plan.md"
    api.prepareWorkbench.mockResolvedValueOnce({ path: disk, cwd: CWD })
    const onMoved = vi.fn()
    const { result, rerender } = renderHook(({ path }) => usePreparedWorkbench(path, onMoved), { initialProps: { path: "project/moved/document/plan.md" } })
    expect(result.current.prepared).toBeNull()
    await waitFor(() => expect(onMoved).toHaveBeenCalledWith(disk))

    rerender({ path: disk })
    expect(result.current.prepared).toEqual({ path: disk, cwd: CWD })
    expect(api.prepareWorkbench).toHaveBeenCalledTimes(1)
  })

  it("prepares once and mounts under StrictMode's double effects", async () => {
    const disk = "/data/workbench/markdown/strict/plan/plan.md"
    api.prepareWorkbench.mockResolvedValueOnce({ path: disk, cwd: CWD })
    const onMoved = vi.fn()
    render(<StrictMode><PreparedItem initialPath="project/strict/document/plan.md" onMoved={onMoved} /></StrictMode>)

    expect(await screen.findByText(`mounted ${disk}`)).toBeInTheDocument()
    expect(api.prepareWorkbench).toHaveBeenCalledTimes(1)
    expect(onMoved).toHaveBeenCalledWith(disk)
  })

  it("keeps a move that finished after its item unmounted", async () => {
    const disk = "/data/workbench/markdown/away/plan/plan.md"
    const answer = deferred()
    api.prepareWorkbench.mockReturnValueOnce(answer.promise)
    const first = render(<PreparedItem initialPath="project/away/document/plan.md" onMoved={vi.fn()} />)
    first.unmount()
    await act(async () => answer.resolve({ path: disk, cwd: CWD }))

    const onMoved = vi.fn()
    render(<PreparedItem initialPath="project/away/document/plan.md" onMoved={onMoved} />)
    expect(await screen.findByText(`mounted ${disk}`)).toBeInTheDocument()
    expect(onMoved).toHaveBeenCalledWith(disk)
    expect(api.prepareWorkbench).toHaveBeenCalledTimes(1)
  })

  it("shows a failure and prepares again on retry", async () => {
    const path = "/data/workbench/markdown/retry/plan/plan.md"
    api.prepareWorkbench.mockRejectedValueOnce(new Error("Name taken"))
    const { result } = renderHook(() => usePreparedWorkbench(path, vi.fn()))
    await waitFor(() => expect(result.current.error).toBe("Name taken"))

    api.prepareWorkbench.mockResolvedValueOnce({ path, cwd: CWD })
    act(() => result.current.retry())
    expect(result.current.error).toBeNull()
    await waitFor(() => expect(result.current.prepared?.cwd).toBe(CWD))
  })

  it("drops an earlier failure when the failed path opens again", async () => {
    const a = "/data/workbench/markdown/again/a/a.md"
    const b = "/data/workbench/markdown/again/b/b.md"
    api.prepareWorkbench.mockRejectedValueOnce(new Error("Not reachable"))
    const { result, rerender } = renderHook(({ path }) => usePreparedWorkbench(path, vi.fn()), { initialProps: { path: a } })
    await waitFor(() => expect(result.current.error).toBe("Not reachable"))

    api.prepareWorkbench.mockResolvedValueOnce({ path: b, cwd: CWD })
    rerender({ path: b })
    expect(result.current.error).toBeNull()
    await waitFor(() => expect(result.current.prepared?.path).toBe(b))

    const answer = deferred()
    api.prepareWorkbench.mockReturnValueOnce(answer.promise)
    rerender({ path: a })
    expect(result.current.error).toBeNull()
    await act(async () => answer.resolve({ path: a, cwd: CWD }))
    expect(result.current.prepared?.path).toBe(a)
    expect(result.current.error).toBeNull()
  })

  it("stays prepared under the path the item was opened with when the file is renamed", async () => {
    const opened = "/data/workbench/markdown/renamed/plan/plan.md"
    const renamed = "/data/workbench/markdown/renamed/plan/roadmap.md"
    api.prepareWorkbench.mockResolvedValueOnce({ path: opened, cwd: CWD })
    const { result, rerender } = renderHook(({ path, sessionKey }) => usePreparedWorkbench(path, vi.fn(), sessionKey), { initialProps: { path: opened, sessionKey: undefined as string | undefined } })
    await waitFor(() => expect(result.current.prepared?.path).toBe(opened))

    rerender({ path: renamed, sessionKey: opened })
    expect(result.current.prepared).toEqual({ path: renamed, cwd: CWD })
    expect(result.current.error).toBeNull()
    expect(api.prepareWorkbench).toHaveBeenCalledTimes(1)
  })

  it("stays prepared for the next file of a session that was prepared under another path", async () => {
    const session = "/home/me/thesis/main.tex"
    const restored = "/home/me/thesis/chapters/intro.tex"
    const next = "/home/me/thesis/chapters/outro.tex"
    api.prepareWorkbench.mockResolvedValueOnce({ path: restored, cwd: CWD })
    const { result, rerender } = renderHook(({ path }) => usePreparedWorkbench(path, vi.fn(), session), { initialProps: { path: restored } })
    await waitFor(() => expect(result.current.prepared?.path).toBe(restored))

    rerender({ path: next })
    expect(result.current.prepared).toEqual({ path: next, cwd: CWD })
    expect(api.prepareWorkbench).toHaveBeenCalledTimes(1)
  })
})

describe("useFollowBuffers", () => {
  const OTHER = "/data/workbench/markdown/proj/plan/notes.md"
  const synced = (entries: Record<string, string>) => ({ current: new Map(Object.entries(entries)) as ReadonlyMap<string, string> })

  it("follows the open files and pushes a changed one into Neovim when its disk text differs from the synced text", async () => {
    const setContent = vi.fn(() => true)
    const buffers = synced({ [PATH]: "old", [OTHER]: "same" })
    renderHook(() => useFollowBuffers({ synced: buffers, paths: [PATH, OTHER], setContent }))
    expect(follow.paths).toEqual([PATH, OTHER])

    api.loadFileViewerText.mockResolvedValueOnce("remote")
    act(() => follow.changed?.(OTHER))
    await waitFor(() => expect(setContent).toHaveBeenCalledExactlyOnceWith("remote", OTHER))

    // Text equal to the synced text and a file that is not open push nothing
    api.loadFileViewerText.mockResolvedValue("old")
    act(() => follow.changed?.(PATH))
    act(() => follow.changed?.("/data/closed.md"))
    await waitFor(() => expect(api.loadFileViewerText).toHaveBeenCalledTimes(3))
    await act(async () => {})
    expect(setContent).toHaveBeenCalledTimes(1)
  })

  it("pushes again on the next change when Neovim could not take a push and skips a file closed while it was read", async () => {
    const buffers = synced({ [PATH]: "old", [OTHER]: "old notes" })
    const setContent = vi.fn().mockReturnValueOnce(false).mockReturnValue(true)
    renderHook(() => useFollowBuffers({ synced: buffers, paths: [PATH, OTHER], setContent }))

    api.loadFileViewerText.mockResolvedValue("remote")
    act(() => follow.changed?.(PATH))
    await waitFor(() => expect(setContent).toHaveBeenCalledTimes(1))
    act(() => follow.changed?.(PATH))
    await waitFor(() => expect(setContent).toHaveBeenCalledTimes(2))

    const gate = Promise.withResolvers<string>()
    api.loadFileViewerText.mockReturnValueOnce(gate.promise)
    act(() => follow.changed?.(OTHER))
    buffers.current = new Map([[PATH, "remote"]])
    await act(async () => { gate.resolve("late") })
    expect(setContent).toHaveBeenCalledTimes(2)
  })

  it("keeps quiet when the file cannot be read", async () => {
    const setContent = vi.fn(() => true)
    renderHook(() => useFollowBuffers({ synced: synced({ [PATH]: "old" }), paths: [PATH], setContent }))

    api.loadFileViewerText.mockRejectedValueOnce(new Error("gone"))
    await act(async () => { follow.changed?.(PATH) })
    expect(setContent).not.toHaveBeenCalled()
  })
})
