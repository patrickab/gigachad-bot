import { describe, expect, it } from "vitest"
import { reframeCamera, screenToWorld, zoomCamera, type InfiniteViewportCamera, type ViewportSize } from "@/components/InfiniteViewport"

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

  it("keeps the center and magnifies the view when an embedded frame follows its host's zoom", () => {
    const offset = { x: 130, y: -40 }
    const scale = 0.8
    const next = { width: viewport.width * 1.5, height: viewport.height * 1.5 }
    const before = { x: (viewport.width / 2 - offset.x) / scale, y: (viewport.height / 2 - offset.y) / scale }

    const view = reframeCamera(offset, scale, viewport, next, 1.5)
    const after = { x: (next.width / 2 - view.offset.x) / view.scale, y: (next.height / 2 - view.offset.y) / view.scale }

    expect(view.scale).toBeCloseTo(1.2)
    expect(after.x).toBeCloseTo(before.x)
    expect(after.y).toBeCloseTo(before.y)
    // Same world span across the frame: the view is unchanged, only magnified.
    expect(next.width / view.scale).toBeCloseTo(viewport.width / scale)
  })
})
