import { describe, expect, it } from "vitest"
import { parsePatch } from "@/lib/diff"

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
