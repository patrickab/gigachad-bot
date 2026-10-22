import { describe, expect, it } from "vitest"
import { parsePatch, toSplitRows } from "@/lib/diff"

describe("parsePatch", () => {
  it("numbers lines per side from the hunk header and drops the file header", () => {
    const patch = ["diff --git a/f b/f", "index 1..2", "--- a/f", "+++ b/f", "@@ -10,3 +20,3 @@ fn", " keep", "-old", "+new", "\\ No newline at end of file"].join("\n")

    expect(parsePatch(patch).map(({ kind, text, oldNumber, newNumber }) => [kind, text, oldNumber, newNumber])).toEqual([
      ["hunk", "@@ -10,3 +20,3 @@ fn", null, null],
      ["context", "keep", 10, 20],
      ["removed", "old", 11, null],
      ["added", "new", null, 21],
      ["note", "\\ No newline at end of file", null, null],
    ])
  })
})

describe("toSplitRows", () => {
  it("pairs a removed run with the added run after it, leftovers face an empty side", () => {
    const lines = parsePatch(["@@ -1,4 +1,3 @@", " keep", "-a", "-b", "-c", "+x", "+y", " tail", "+only"].join("\n"))

    expect(toSplitRows(lines)).toEqual([
      { full: 0 },
      { old: 1, new: 1 },
      { old: 2, new: 5 },
      { old: 3, new: 6 },
      { old: 4, new: null },
      { old: 7, new: 7 },
      { old: null, new: 8 },
    ])
  })
})
