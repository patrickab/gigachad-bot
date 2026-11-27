import { afterEach, describe, expect, it } from "vitest"
import { clearOutbox, confirmedIds, hold } from "../outbox"

const noMessages = () => []
const none = { steering: [], followUp: [] }

describe("outbox", () => {
  afterEach(clearOutbox)

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
