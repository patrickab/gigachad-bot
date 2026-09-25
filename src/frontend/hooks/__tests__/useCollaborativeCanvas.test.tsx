import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { CanvasDocument } from "@/components/CanvasEditor"
import type { CanvasAcceptedBatch, CanvasMutation } from "@/lib/api"

const api = vi.hoisted(() => {
  class MockApiError extends Error {
    constructor(message: string, readonly status: number) {
      super(message)
    }
  }
  return { ApiError: MockApiError, loadCanvasSnapshot: vi.fn(), submitCanvasMutations: vi.fn() }
})
const stream = vi.hoisted(() => ({ subscribeToCanvasMutationBatches: vi.fn() }))

vi.mock("@/lib/api", () => api)
vi.mock("@/lib/canvasStream", () => stream)

import { useCollaborativeCanvas, type CollaborativeCanvas } from "@/hooks/useCollaborativeCanvas"

const KEY = "project/board.canvas"
const EMPTY: CanvasDocument = { version: 1, frames: [], strokes: [], attachments: [], texts: [] }
const page = (id: string, x = 0) => ({ id, kind: "page" as const, x, y: 0, width: 794 })

async function flush() {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
}

describe("useCollaborativeCanvas", () => {
  let deliver: (batch: CanvasAcceptedBatch) => void
  let streamError: () => void
  let streamOpen: () => void

  const sent = (call: number) => api.submitCanvasMutations.mock.calls[call]![1] as CanvasMutation[]
  const batch = (revision: number, mutations: CanvasMutation[]) => act(() => deliver({ canvasKey: KEY, revision, mutations }))

  async function edit(result: { current: CollaborativeCanvas }, frames: CanvasDocument["frames"]) {
    act(() => result.current.replace({ ...result.current.document!, frames }))
    act(() => vi.advanceTimersByTime(75))
    await flush()
  }

  beforeEach(() => {
    vi.useFakeTimers()
    api.loadCanvasSnapshot.mockReset().mockResolvedValue({ revision: 0, document: EMPTY })
    api.submitCanvasMutations.mockReset().mockReturnValue(Promise.withResolvers().promise)
    stream.subscribeToCanvasMutationBatches.mockReset().mockImplementation(
      (_key: string, _since: () => number, onBatch: typeof deliver, onError: () => void, onOpen: () => void) => {
        deliver = onBatch
        streamError = onError
        streamOpen = onOpen
        return vi.fn()
      },
    )
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("keeps an acknowledged batch visible until the stream delivers it, then sends queued edits", async () => {
    api.submitCanvasMutations.mockResolvedValueOnce({ revision: 1 }).mockResolvedValueOnce({ revision: 2 })
    const { result } = renderHook(() => useCollaborativeCanvas(KEY))
    await flush()

    await edit(result, [page("first")])
    await edit(result, [page("first"), page("second")])
    expect(api.submitCanvasMutations).toHaveBeenCalledTimes(1)
    expect(result.current.document!.frames.map((frame) => frame.id)).toEqual(["first", "second"])
    expect(result.current.status).toBe("saving")

    batch(1, sent(0))
    await flush()
    expect(api.submitCanvasMutations).toHaveBeenCalledTimes(2)
    expect(sent(1).map((change) => change.id)).toEqual(["second"])

    batch(2, sent(1))
    expect(result.current.document!.frames.map((frame) => frame.id)).toEqual(["first", "second"])
    expect(result.current.status).toBe("synced")
  })

  it("rebases pending local edits over remote batches", async () => {
    const { result } = renderHook(() => useCollaborativeCanvas(KEY))
    await flush()

    await edit(result, [page("local")])
    batch(1, [{ mutationId: "remote-1", kind: "upsert", entity: "frame", id: "remote", value: page("remote", 900) }])

    expect(result.current.document!.frames.map((frame) => frame.id).sort()).toEqual(["local", "remote"])
  })

  it("retries a network failure with the original mutation IDs", async () => {
    api.submitCanvasMutations.mockRejectedValueOnce(new TypeError("network unavailable")).mockResolvedValueOnce({ revision: 1 })
    const { result } = renderHook(() => useCollaborativeCanvas(KEY))
    await flush()

    await edit(result, [page("first")])
    expect(result.current.status).toBe("error")
    act(() => vi.advanceTimersByTime(250))
    await flush()

    expect(sent(1).map((change) => change.mutationId)).toEqual(sent(0).map((change) => change.mutationId))
    batch(1, sent(1))
    expect(result.current.status).toBe("synced")
  })

  it("does not retry a rejected mutation or a malformed canvas", async () => {
    api.submitCanvasMutations.mockRejectedValueOnce(new api.ApiError("invalid mutation", 422))
    const { result } = renderHook(() => useCollaborativeCanvas(KEY))
    await flush()
    await edit(result, [page("first")])
    act(() => vi.advanceTimersByTime(60_000))
    expect(api.submitCanvasMutations).toHaveBeenCalledTimes(1)
    expect(result.current.status).toBe("error")

    api.loadCanvasSnapshot.mockRejectedValueOnce(new api.ApiError("Canvas snapshot is not valid JSON", 422))
    const malformed = renderHook(() => useCollaborativeCanvas("project/broken.canvas"))
    await flush()
    act(() => vi.advanceTimersByTime(60_000))
    expect(api.loadCanvasSnapshot).toHaveBeenCalledTimes(2)
    expect(malformed.result.current).toMatchObject({ document: null, status: "error" })
  })

  it("recovers a transient snapshot failure and clears stream errors on reconnect", async () => {
    api.loadCanvasSnapshot.mockRejectedValueOnce(new TypeError("network unavailable"))
    const { result } = renderHook(() => useCollaborativeCanvas(KEY))
    await flush()
    expect(result.current).toMatchObject({ document: null, status: "error" })

    act(() => vi.advanceTimersByTime(250))
    await flush()
    expect(result.current.document).toEqual(EMPTY)

    act(() => streamError())
    expect(result.current.status).toBe("error")
    act(() => streamOpen())
    expect(result.current.status).toBe("synced")
  })
})
