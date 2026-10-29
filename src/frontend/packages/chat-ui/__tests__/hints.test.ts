import { describe, expect, it } from "vitest"
import { hintLabels } from "../src/hints"

describe("hintLabels", () => {
  // A label that prefixes another would activate its target before the longer one could be typed.
  it("stays unique and prefix-free on both sides of the single-letter limit", () => {
    for (const count of [1, 3, 4, 7, 20]) {
      const labels = hintLabels(count, "abc")
      expect(labels).toHaveLength(count)
      expect(new Set(labels).size).toBe(count)
      for (const label of labels) {
        expect(label).not.toBe("")
        expect(labels.filter((other) => other !== label && other.startsWith(label))).toEqual([])
      }
    }
  })

  it("uses single letters while they suffice", () => {
    expect(hintLabels(3, "abc")).toEqual(["a", "b", "c"])
  })
})
