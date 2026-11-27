import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { getStroke } from "perfect-freehand"
import { renderCanvasToJpeg } from "@/lib/canvasExport"
import type { StrokeData } from "@/lib/drawing"

vi.mock("perfect-freehand", async (importOriginal) => {
  const original = await importOriginal<typeof import("perfect-freehand")>()
  return { getStroke: vi.fn(original.getStroke) }
})

describe("canvas export", () => {
  // jsdom has no 2D canvas: a context that takes every call is enough to count the outlines computed.
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(new Proxy({}, { get: () => () => {} }) as never)
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation((done) => done(new Blob()))
    vi.stubGlobal("Path2D", class {})
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("reuses the outlines of strokes it has already drawn", async () => {
    const strokes: StrokeData[] = Array.from({ length: 2000 }, (_, i) => ({
      id: `s${i}`, color: "#000", width: 2, points: [[i, 0, 0.5], [i + 5, 5, 0.5], [i + 10, 0, 0.5]],
    }))
    await renderCanvasToJpeg(strokes)
    vi.mocked(getStroke).mockClear()

    await renderCanvasToJpeg(strokes)

    expect(getStroke).toHaveBeenCalledTimes(0)
  })
})
