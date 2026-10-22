import { describe, expect, it } from "vitest"
import { parsePatch } from "@/lib/diff"
import { anchorAt, formatReview, locateNote } from "@/features/editor/diff/review"

const lines = parsePatch(["@@ -1,3 +1,3 @@", " one", "-two", "+TWO", " three", "@@ -9,2 +9,2 @@", " nine", "+ten"].join("\n"))

describe("review notes", () => {
  it("quotes up to two neighbours on each side without crossing into the next hunk", () => {
    expect(anchorAt("f.ts", lines, 4, "new")).toMatchObject({ path: "f.ts", side: "new", line: 3, lineText: "three", quote: ["-two", "+TWO", " three"] })
    expect(anchorAt("f.ts", lines, 2, "old")).toMatchObject({ side: "old", line: 2, quote: [" one", "-two", "+TWO", " three"] })
    expect(anchorAt("f.ts", lines, 0, "new")).toBeNull()
  })

  it("follows its line when lines are inserted above, and reports it gone once the text changed", () => {
    const anchor = anchorAt("f.ts", lines, 3, "new")!
    const shifted = parsePatch(["@@ -1,3 +1,5 @@", "+a", "+b", " one", "-two", "+TWO", " three"].join("\n"))
    expect(shifted[locateNote(anchor, shifted)!]).toMatchObject({ text: "TWO", newNumber: 4 })
    expect(locateNote(anchor, parsePatch(["@@ -1,3 +1,3 @@", " one", "-two", "+Two", " three"].join("\n")))).toBeNull()
  })

  it("orders the sent review by file and line, with each note's quote", () => {
    const later = { ...anchorAt("b.ts", lines, 4, "new")!, id: "1", comment: "rename" }
    const earlier = { ...anchorAt("a.ts", lines, 2, "old")!, id: "2", comment: "keep this" }
    const text = formatReview([later, earlier], { "a.ts": lines, "b.ts": lines })
    expect(text.indexOf("1. `a.ts:2` (old side)")).toBeLessThan(text.indexOf("2. `b.ts:3`"))
    expect(text).toContain("```diff\n-two\n+TWO\n three\n```\nrename")
  })

  it("sends the resolved location and current quote after lines shift", () => {
    const note = { ...anchorAt("f.ts", lines, 3, "new")!, id: "n", comment: "fix this" }
    const shifted = parsePatch(["@@ -1,3 +1,4 @@", "+inserted", " one", "-two", "+TWO", " three"].join("\n"))
    const text = formatReview([note], { "f.ts": shifted })
    expect(text).toContain("`f.ts:3`")
    expect(text).not.toContain("outdated")
  })

  it("does not attach a replaced target to an unrelated duplicate", () => {
    const original = parsePatch("@@ -1,3 +1,3 @@\n before\n+target\n after")
    const note = { ...anchorAt("f.ts", original, 2, "new")!, id: "n", comment: "fix this" }
    const changed = parsePatch("@@ -1,3 +1,5 @@\n before\n+changed\n after\n+unrelated\n+target")
    expect(locateNote(note, changed)).toBeNull()
    expect(formatReview([note], { "f.ts": changed })).toContain("`f.ts:2` (outdated, original location)")
    expect(formatReview([note], {})).toContain("outdated, original location")
  })

  it("uses context to distinguish duplicates and leaves equal matches unresolved", () => {
    const original = parsePatch("@@ -1,3 +1,3 @@\n before\n+target\n after")
    const anchor = anchorAt("f.ts", original, 2, "new")!
    const distinct = parsePatch("@@ -1,3 +1,6 @@\n+other\n+target\n+other\n before\n+target\n after")
    expect(distinct[locateNote(anchor, distinct)!]!.newNumber).toBe(5)
    const ambiguous = parsePatch("@@ -1,6 +1,6 @@\n before\n+target\n after\n before\n+target\n after")
    expect(locateNote(anchor, ambiguous)).toBeNull()
  })
})
