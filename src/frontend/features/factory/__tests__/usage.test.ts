import { describe, expect, it } from "vitest"
import { formatFlow } from "@/features/factory/usage"

describe("formatFlow", () => {
  it("shows input, output and cost", () => {
    expect(formatFlow({ input: 55_000, output: 12_000, cost: 3.5 })).toBe("↑55k ↓12k · $3.50")
  })

  it("counts cache reads as input", () => {
    expect(formatFlow({ input: 5000, output: 12_000, cost: 3.5, cacheRead: 50_000 })).toBe("↑55k ↓12k · $3.50")
  })
})
