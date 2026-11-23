import { afterEach, describe, expect, it } from "vitest"
import { clearOutbox, confirmedIds, heldOf, hold, patchHeld, removeHeld, takeHeld } from "../outbox"

const noMessages = () => []
const none = { steering: [], followUp: [] }

describe("outbox", () => {
  afterEach(clearOutbox)

  it("holds messages oldest first per session", () => {
    hold("a", { text: "one", images: [], mode: "steer" })
    hold("a", { text: "two", images: [], mode: "followUp" })
    hold("b", { text: "other", images: [], mode: "steer" })
    expect(heldOf("a").map((entry) => [entry.text, entry.state, entry.dispatched])).toEqual([["one", "sending", false], ["two", "sending", false]])
    expect(heldOf("b")).toHaveLength(1)
  })

  it("patches, removes and takes", () => {
    const first = hold("a", { text: "one", images: [], mode: "steer" })
    hold("a", { text: "two", images: [], mode: "steer" })
    patchHeld("a", first.id, { state: "failed" })
    expect(heldOf("a")[0].state).toBe("failed")
    removeHeld("a", [first.id])
    expect(takeHeld("a").map((entry) => entry.text)).toEqual(["two"])
    expect(heldOf("a")).toEqual([])
  })

  it("confirms by queue snapshot, one item per entry", () => {
    const entries = [hold("a", { text: "x", images: [], mode: "steer" }), hold("a", { text: "x", images: [], mode: "steer" })]
    expect(confirmedIds(entries, { steering: ["x"], followUp: [] }, noMessages)).toEqual([entries[0].id])
  })

  it("confirms by a user message in the session", () => {
    const entry = hold("a", { text: "hi", images: [], mode: "followUp" })
    expect(confirmedIds([entry], none, () => ["hi"])).toEqual([entry.id])
    expect(confirmedIds([entry], none, () => ["other"])).toEqual([])
  })
})
