import { describe, expect, it } from "vitest"
import { embeddedViewportSize, nestedEmbeddingScale, screenToWorld, zoomCamera, type InfiniteViewportCamera, type ViewportSize } from "@/components/InfiniteViewport"

const viewport: ViewportSize = { width: 800, height: 600 }


describe("InfiniteViewport", () => {
  it("keeps the viewport center fixed for a center-anchored nested zoom", () => {
    const camera: InfiniteViewportCamera = { scale: 0.5, centerX: 240, centerY: -80 }
    const before = screenToWorld({ x: viewport.width / 2, y: viewport.height / 2 }, camera, viewport)
    const next = zoomCamera(camera, 1.5, viewport, "viewport-center")

    const after = screenToWorld({ x: viewport.width / 2, y: viewport.height / 2 }, next, viewport)
    expect(after.x).toBeCloseTo(before.x)
    expect(after.y).toBeCloseTo(before.y)
  })

  it("keeps the world point under the cursor fixed for root zoom", () => {
    const camera: InfiniteViewportCamera = { scale: 0.5, centerX: 240, centerY: -80 }
    const cursor = { x: 120, y: 440 }
    const before = screenToWorld(cursor, camera, viewport)
    const next = zoomCamera(camera, 1.5, viewport, "cursor", cursor)

    const after = screenToWorld(cursor, next, viewport)
    expect(after.x).toBeCloseTo(before.x)
    expect(after.y).toBeCloseTo(before.y)
  })

  it("counteracts parent scale for screen-stable nested viewports", () => {
    const size = { width: 600, height: 420 }

    expect(embeddedViewportSize(size, 2, "screen-stable")).toEqual(size)
    expect(embeddedViewportSize(size, 2, "inherit")).toEqual({ width: 1200, height: 840 })
    expect(nestedEmbeddingScale(undefined)).toBe("screen-stable")
  })
})
