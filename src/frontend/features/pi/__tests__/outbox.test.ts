import { afterEach, describe, expect, it } from "vitest"
import { clearOutbox, confirmFromTranscript, heldOf, hold, takeBack } from "../outbox"

const none = { steering: [], followUp: [] }
const user = (content: string) => ({ role: "user" as const, content, timestamp: Date.now() })

describe("outbox", () => {
  afterEach(clearOutbox)

  it("confirms by queue snapshot, one item per entry", () => {
    hold("a", { text: "x", images: [], mode: "steer" })
    const second = hold("a", { text: "x", images: [], mode: "steer" })
    confirmFromTranscript("a", { messages: [], queue: { steering: ["x"], followUp: [] } })
    expect(heldOf("a")).toEqual([second])
  })

  it("confirms by a user message in the session, one entry per message", () => {
    hold("a", { text: "hi", images: [], mode: "followUp" })
    const other = hold("a", { text: "other", images: [], mode: "followUp" })
    const again = hold("a", { text: "hi", images: [], mode: "followUp" })
    confirmFromTranscript("a", { messages: [user("hi")], queue: none })
    expect(heldOf("a")).toEqual([other, again])
  })

  it("takes back the cleared queue, then the held texts it does not hold, duplicates counted", () => {
    const taken = ["x", "x", "y"].map((text) => hold("a", { text, images: [], mode: "steer" }))
    expect(takeBack(taken, { steering: ["x"], followUp: ["z"] })).toEqual(["x", "z", "x", "y"])
  })
})
