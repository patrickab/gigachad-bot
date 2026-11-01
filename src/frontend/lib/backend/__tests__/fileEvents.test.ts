import { afterEach, describe, expect, it, vi } from "vitest"
import type { ChangeEvent } from "../syncStream"

const feed = vi.hoisted(() => ({ listeners: new Set<(event: ChangeEvent) => void>() }))
vi.mock("../syncStream", () => ({
  subscribeToChanges: (listener: (event: ChangeEvent) => void) => {
    feed.listeners.add(listener)
    return () => { feed.listeners.delete(listener) }
  },
}))

import { announceFileChange, announceFileMove, onFileChange, onFileMove } from "../fileEvents"

const emit = (key: string, movedTo?: string) => {
  const event: ChangeEvent = { seq: 1, resource_kind: "document", resource_key: key, moved_to: movedTo, version: null, device_id: "other" }
  for (const listener of feed.listeners) listener(event)
}

afterEach(() => { feed.listeners.clear() })

describe("onFileMove", () => {
  it("hears this device's announcement and another device's moved_to event, until it unsubscribes", () => {
    const listener = vi.fn()
    const unsubscribe = onFileMove(listener)

    announceFileMove({ from: "/a.tex", to: "/b.tex" })
    emit("/c.tex", "/d.tex")
    emit("/e.tex")
    expect(listener.mock.calls).toEqual([[{ from: "/a.tex", to: "/b.tex" }], [{ from: "/c.tex", to: "/d.tex" }]])

    unsubscribe()
    announceFileMove({ from: "/f.tex", to: "/g.tex" })
    expect(listener).toHaveBeenCalledTimes(2)
    expect(feed.listeners.size).toBe(0)
  })
})

describe("onFileChange", () => {
  it("hears the content changes of the files it matches and no moves", () => {
    const exact = vi.fn()
    const predicate = vi.fn()
    onFileChange("/a.tex", exact)
    onFileChange((path) => path.endsWith(".bib"), predicate)

    emit("/a.tex")
    emit("/a.tex", "/b.tex")
    emit("/refs.bib")
    expect(exact.mock.calls).toEqual([["/a.tex"]])
    expect(predicate.mock.calls).toEqual([["/refs.bib"]])
  })

  it("hears this device's announcement only where the match accepts it, until it unsubscribes", () => {
    const exact = vi.fn()
    const other = vi.fn()
    const unsubscribe = onFileChange("/a.tex", exact)
    onFileChange("/b.tex", other)

    announceFileChange("/a.tex")
    expect(exact.mock.calls).toEqual([["/a.tex"]])
    expect(other).not.toHaveBeenCalled()

    unsubscribe()
    announceFileChange("/a.tex")
    expect(exact).toHaveBeenCalledTimes(1)
  })
})
