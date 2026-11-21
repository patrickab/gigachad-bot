import { describe, expect, it } from "vitest"
import type { ReviewNote } from "@/features/editor/diff/review"
import { formatTaskReview } from "@/features/factory/taskReview"

const note = (path: string, line: number, lineText: string, comment: string): ReviewNote =>
  ({ id: `${path}:${line}`, path, side: "new", line, lineText, quote: [`+${lineText}`], comment })

describe("formatTaskReview", () => {
  it("writes a first line, then path:line, the quoted line fenced and the comment per comment, in path and line order", () => {
    expect(formatTaskReview([note("src/b.ts", 3, "const b = 2", "Inline it"), note("src/a.ts", 12, "  return x", "Guard null\nfirst")])).toBe([
      "Review of the diff:",
      "src/a.ts:12\n```\n  return x\n```\nGuard null\nfirst",
      "src/b.ts:3\n```\nconst b = 2\n```\nInline it",
    ].join("\n\n"))
  })

  it("fences a quoted line that holds backticks with a longer fence", () => {
    expect(formatTaskReview([note("a.md", 1, "```ts", "Close it")])).toBe("Review of the diff:\n\na.md:1\n````\n```ts\n````\nClose it")
  })
})
