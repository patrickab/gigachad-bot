import { describe, expect, it } from "vitest"
import { strokesInLasso } from "@/features/editor/canvas/selection/canvasSelection"
import type { StrokeData } from "@/lib/drawing"

describe("strokesInLasso", () => {
  const stroke = (id: string, x: number): StrokeData => ({ id, points: [[x, 5], [x + 1, 5]], color: "#000", width: 2 })
  const loop: [number, number][] = [[0, 0], [20, 0], [20, 10], [0, 10]]

  it("selects strokes by id, mostly-covered strokes only", () => {
    expect([...strokesInLasso([stroke("in", 5), stroke("out", 50), stroke("half", 19.5)], loop)]).toEqual(["in"])
  })
})
