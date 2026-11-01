import { StrictMode, useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render, renderHook, screen, waitFor } from "@testing-library/react"
import type { ChangeEvent } from "@/lib/backend/syncStream"

const api = vi.hoisted(() => ({
  loadFileViewerText: vi.fn<(path: string) => Promise<string>>(),
  prepareWorkbench: vi.fn<(path: string) => Promise<{ path: string; cwd: string }>>(),
}))
vi.mock("@/lib/backend/api", () => api)
const feed = vi.hoisted(() => ({ listener: null as ((event: ChangeEvent) => void) | null }))
vi.mock("@/lib/backend/syncStream", () => ({
  subscribeToChanges: (listener: (event: ChangeEvent) => void) => {
    feed.listener = listener
    return () => { feed.listener = null }
  },
}))

import { useAgentSync } from "../useAgentSync"
import { usePreparedWorkbench } from "../usePreparedWorkbench"

const PATH = "/data/workbench/markdown/proj/plan/plan.md"
const CWD = "/data/workbench/markdown"
const change = (key: string): ChangeEvent => ({ seq: 1, resource_kind: "document", resource_key: key, version: null, device_id: "other" })

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
})

describe("useAgentSync", () => {
  it("pushes another device's save of the file into Neovim when it differs from the synced text", async () => {
    const setContent = vi.fn(() => true)
    const syncedRef = { current: "old" as string | null }
    renderHook(() => useAgentSync({ path: PATH, streaming: false, setContent, syncedRef, onRunEnd: async () => {} }))

    api.loadFileViewerText.mockResolvedValueOnce("remote")
    act(() => feed.listener?.(change(PATH)))
    await waitFor(() => expect(setContent).toHaveBeenCalledWith("remote"))
    expect(syncedRef.current).toBe("remote")

    api.loadFileViewerText.mockResolvedValueOnce("remote")
    act(() => feed.listener?.(change(PATH)))
    act(() => feed.listener?.(change("/data/other.md")))
    await waitFor(() => expect(api.loadFileViewerText).toHaveBeenCalledTimes(2))
    expect(setContent).toHaveBeenCalledTimes(1)
  })
})
