import { afterEach, describe, expect, it, vi } from "vitest"
import { renderCanvasToJpeg } from "@/lib/drawing"

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("renderCanvasToJpeg", () => {
  it("uses the source image dimensions instead of a stale frame aspect", async () => {
    const context = {
      scale: vi.fn(),
      fillStyle: "",
      fillRect: vi.fn(),
      drawImage: vi.fn(),
    }
    // CanvasRenderingContext2D has many unrelated methods that this stroke-free test never calls.
    const canvasContext = context as unknown as CanvasRenderingContext2D
    const sizes: number[][] = []
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(canvasContext)
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (this: HTMLCanvasElement, callback) {
      sizes.push([this.width, this.height])
      callback(new Blob())
    })
    vi.stubGlobal("fetch", vi.fn(async () => new Response(new Blob(["image"]))))
    // Only width and height are read from the browser ImageBitmap in this test.
    const bitmap = { width: 120, height: 60 } as unknown as ImageBitmap
    vi.stubGlobal("createImageBitmap", vi.fn(async () => bitmap))

    await renderCanvasToJpeg([], 20, [{ url: "/image.png", x: 10, y: 20, width: 120, aspect: 1.3 }])

    expect(context.drawImage).toHaveBeenCalledWith(bitmap, 20, 20, 120, 60)
    expect(sizes).toEqual([[320, 200]])
  })
})
