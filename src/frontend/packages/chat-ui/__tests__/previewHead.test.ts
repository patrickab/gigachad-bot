import { describe, expect, it } from "vitest"
import { previewHead } from "../src/previewHead.ts"

describe("previewHead", () => {
  it("returns short markdown unchanged", () => {
    expect(previewHead("a\n\nb", 100)).toBe("a\n\nb")
  })

  it("cuts at the first blank line past the limit", () => {
    const text = `${"a".repeat(30)}\n\n${"b".repeat(30)}\n\n${"c".repeat(30)}`
    expect(previewHead(text, 40)).toBe(`${"a".repeat(30)}\n\n${"b".repeat(30)}\n`)
  })

  it("never cuts inside a code fence or a display equation", () => {
    const fence = `intro\n\n\`\`\`py\n${"x = 1\n\n".repeat(5)}\`\`\`\n\nafter\n\ntail`
    expect(previewHead(fence, 30)).toBe(fence.slice(0, fence.indexOf("after") - 1))
    const math = `intro\n\n$$\n${"a + b\n\n".repeat(5)}$$\n\nafter\n\ntail`
    expect(previewHead(math, 30)).toBe(math.slice(0, math.indexOf("after") - 1))
  })

  it("cuts a block without an end at a line end, four limits in", () => {
    const head = previewHead(`\`\`\`\n${"line\n".repeat(200)}`, 50)
    expect(head.length).toBeLessThan(260)
    expect(head.endsWith("line\n")).toBe(true)
  })
})
