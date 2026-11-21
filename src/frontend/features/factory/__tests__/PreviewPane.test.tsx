import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import type { FactoryPreview } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({
  getFactoryPreview: vi.fn<(id: string) => Promise<FactoryPreview | null>>(),
  startFactoryPreview: vi.fn<(id: string) => Promise<FactoryPreview>>(),
  stopFactoryPreview: vi.fn<(id: string) => Promise<FactoryPreview>>(),
}))

vi.mock("@/lib/backend/api", () => api)

import { PREVIEW_POLL_MS, PreviewPane } from "@/features/factory/PreviewPane"

const STARTING: FactoryPreview = { state: "starting", url: null, error: null }
const READY: FactoryPreview = { state: "ready", url: "http://gate.test/p/8100/", error: null }
const STOPPED: FactoryPreview = { state: "stopped", url: null, error: null }
const FAILED: FactoryPreview = { state: "failed", url: null, error: "npm ERR! missing script: dev" }

/** Run pending promise callbacks, advancing the fake clock by `ms`. */
async function advance(ms: number) {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms) })
}

beforeEach(() => {
  vi.useFakeTimers()
  api.getFactoryPreview.mockReset().mockResolvedValue(null)
  api.startFactoryPreview.mockReset().mockResolvedValue(STARTING)
  api.stopFactoryPreview.mockReset().mockResolvedValue(STOPPED)
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("PreviewPane", () => {
  it("offers to start a preview that was never started or is stopped, without polling", async () => {
    render(<PreviewPane taskId="t1" />)
    await advance(0)
    expect(screen.getByRole("button", { name: "Start preview" })).toBeTruthy()

    await advance(PREVIEW_POLL_MS * 3)
    expect(api.getFactoryPreview).toHaveBeenCalledTimes(1)

    cleanup()
    api.getFactoryPreview.mockResolvedValue(STOPPED)
    render(<PreviewPane taskId="t1" />)
    await advance(0)
    expect(screen.getByRole("button", { name: "Start preview" })).toBeTruthy()
  })

  it("polls a starting preview every interval until it is ready, then shows the frame and stops polling", async () => {
    render(<PreviewPane taskId="t1" />)
    await advance(0)
    fireEvent.click(screen.getByRole("button", { name: "Start preview" }))
    await advance(0)
    expect(api.startFactoryPreview).toHaveBeenCalledWith("t1")
    expect(screen.getByRole("status").textContent).toBe("Starting the preview")
    expect(api.getFactoryPreview).toHaveBeenCalledTimes(1)

    api.getFactoryPreview.mockResolvedValueOnce(STARTING).mockResolvedValueOnce(READY)
    await advance(PREVIEW_POLL_MS)
    expect(api.getFactoryPreview).toHaveBeenCalledTimes(2)
    expect(screen.queryByTitle("Task preview")).toBeNull()

    await advance(PREVIEW_POLL_MS)
    expect(api.getFactoryPreview).toHaveBeenCalledTimes(3)
    expect(screen.getByTitle("Task preview").getAttribute("src")).toBe(READY.url)

    await advance(PREVIEW_POLL_MS * 3)
    expect(api.getFactoryPreview).toHaveBeenCalledTimes(3)
  })

  it("keeps polling a starting preview after a failed read", async () => {
    api.getFactoryPreview.mockResolvedValueOnce(STARTING).mockRejectedValueOnce(new Error("backend restarting")).mockResolvedValueOnce(READY)
    render(<PreviewPane taskId="t1" />)
    await advance(0)
    await advance(PREVIEW_POLL_MS)
    expect(screen.getByRole("alert").textContent).toBe("backend restarting")

    await advance(PREVIEW_POLL_MS)
    expect(screen.getByTitle("Task preview")).toBeTruthy()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("stops a ready preview and offers to start it again", async () => {
    api.getFactoryPreview.mockResolvedValue(READY)
    render(<PreviewPane taskId="t1" />)
    await advance(0)
    fireEvent.click(screen.getByRole("button", { name: "Stop preview" }))
    await advance(0)

    expect(api.stopFactoryPreview).toHaveBeenCalledWith("t1")
    expect(screen.queryByTitle("Task preview")).toBeNull()
    expect(screen.getByRole("button", { name: "Start preview" })).toBeTruthy()
  })

  it("shows the error tail of a failed preview and starts it again", async () => {
    api.getFactoryPreview.mockResolvedValue(FAILED)
    render(<PreviewPane taskId="t1" />)
    await advance(0)
    expect(screen.getByLabelText("Preview error").textContent).toBe("npm ERR! missing script: dev")

    await advance(PREVIEW_POLL_MS * 2)
    expect(api.getFactoryPreview).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: "Start preview" }))
    await advance(0)
    expect(screen.getByRole("status").textContent).toBe("Starting the preview")
  })

  it("shows why a start was refused", async () => {
    api.startFactoryPreview.mockRejectedValue(new Error("factory.yaml has no preview"))
    render(<PreviewPane taskId="t1" />)
    await advance(0)
    fireEvent.click(screen.getByRole("button", { name: "Start preview" }))
    await advance(0)

    expect(screen.getByRole("alert").textContent).toBe("factory.yaml has no preview")
    expect(screen.getByRole("button", { name: "Start preview" })).toBeTruthy()
  })
})
