// @vitest-environment node
import { fileURLToPath } from "node:url"
import { beforeAll, describe, expect, it } from "vitest"
import { bendPoint, loadRouter, routeEdges, type Point, type RouteCard } from "@/lib/edgeRouting"

const card = (id: string, x: number, y: number, extra: Partial<RouteCard> = {}): RouteCard =>
  ({ id, x, y, width: 200, height: 100, parent: null, compound: false, ...extra })

/** Whether any segment of the route passes through the inside of `rect`. */
function crosses(route: Point[], rect: RouteCard): boolean {
  return route.slice(1).some((to, index) => {
    const from = route[index]
    return Math.max(from.x, to.x) > rect.x && Math.min(from.x, to.x) < rect.x + rect.width
      && Math.max(from.y, to.y) > rect.y && Math.min(from.y, to.y) < rect.y + rect.height
  })
}

describe("routeEdges", () => {
  beforeAll(async () => {
    expect(await loadRouter(fileURLToPath(new URL("../../public/libavoid.wasm", import.meta.url)))).toBe(true)
  })

  it("routes around cards a connection does not join, including a card right between its ends", () => {
    const cards = [card("top", 0, 0), card("middle", 0, 200), card("bottom", 0, 400)]
    const route = routeEdges(cards, [{ id: "e", source: "top", target: "bottom", bend: 0 }])!.get("e")!
    expect(crosses(route, cards[1])).toBe(false)
    // It leaves and arrives on the cards' borders.
    expect(route[0].y === 100 || route[0].x === 0 || route[0].x === 200).toBe(true)
  })

  it("never runs two connections along the same stretch, even when they cross different frames", () => {
    // The canvas sync view as LikeC4 lays it out: three frames, nine connections.
    const at = (id: string, x: number, y: number, width: number, height: number, compound = false): RouteCard =>
      ({ id, x, y, width, height, compound, parent: id.includes(".") ? id.split(".")[0] : null })
    const cards = [
      at("app", 438, -3, 408, 276, true), at("app.canvasClient", 470, 61, 343, 180),
      at("backend", 439, 329, 981, 284, true), at("backend.broker", 1056, 393, 323, 180), at("backend.canvas", 479, 393, 326, 180),
      at("database", 8, 661, 1268, 607, true), at("database.documents", 48, 725, 320, 180), at("database.mutations", 478, 725, 328, 180),
      at("database.changes", 916, 725, 320, 180), at("database.notify", 916, 1048, 320, 180),
    ]
    const edges = [
      ["app.canvasClient", "backend.canvas"], ["backend.canvas", "app.canvasClient"], ["backend.broker", "backend.canvas"],
      ["backend.canvas", "database.documents"], ["backend.canvas", "database.mutations"], ["backend.canvas", "database.changes"],
      ["backend.broker", "database.changes"], ["database.notify", "backend.broker"], ["database.changes", "database.notify"],
    ].map(([source, target]) => ({ id: `${source}->${target}`, source, target, bend: 0 }))
    const routes = routeEdges(cards, edges)!
    expect(routes.size).toBe(edges.length)
    const segments = (route: Point[]) => route.slice(1).map((to, index) => [route[index], to] as const)
    const shared = (a: readonly [Point, Point], b: readonly [Point, Point]) =>
      (a[0].y === a[1].y && b[0].y === b[1].y && a[0].y === b[0].y
        && Math.min(Math.max(a[0].x, a[1].x), Math.max(b[0].x, b[1].x)) > Math.max(Math.min(a[0].x, a[1].x), Math.min(b[0].x, b[1].x)))
      || (a[0].x === a[1].x && b[0].x === b[1].x && a[0].x === b[0].x
        && Math.min(Math.max(a[0].y, a[1].y), Math.max(b[0].y, b[1].y)) > Math.max(Math.min(a[0].y, a[1].y), Math.min(b[0].y, b[1].y)))
    const all = [...routes.values()]
    all.forEach((a, i) => all.slice(i + 1).forEach((b) => {
      for (const one of segments(a)) for (const other of segments(b)) expect(shared(one, other)).toBe(false)
    }))
  })

  it("keeps parallel connections between the same cards apart", () => {
    const cards = [card("a", 0, 0), card("b", 400, 300)]
    const routes = routeEdges(cards, [{ id: "1", source: "a", target: "b", bend: 0 }, { id: "2", source: "a", target: "b", bend: 0 }])!
    expect(routes.get("1")).not.toEqual(routes.get("2"))
    expect(routes.get("1")![0]).not.toEqual(routes.get("2")![0])
  })

  it("passes a bent connection through its bend point", () => {
    const cards = [card("a", 0, 0), card("b", 600, 0)]
    const through = bendPoint(cards[0], cards[1], 150)
    const route = routeEdges(cards, [{ id: "e", source: "a", target: "b", bend: 150 }])!.get("e")!
    const onRoute = route.slice(1).some((to, index) => {
      const from = route[index]
      return Math.min(from.x, to.x) <= through.x && through.x <= Math.max(from.x, to.x) && Math.min(from.y, to.y) <= through.y && through.y <= Math.max(from.y, to.y)
    })
    expect(onRoute).toBe(true)
  })
})
