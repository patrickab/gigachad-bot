import { describe, expect, it } from "vitest"
import { parsePatch, parsePatchFiles, toSplitRows } from "@/lib/diff"

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

describe("parsePatchFiles", () => {
  it("splits files at their headers, keeping paths without timestamps", () => {
    const patch = ["diff --git a/a.ts b/a.ts", "--- a/a.ts\t2026-01-01", "+++ b/a.ts", "@@ -1 +1 @@", "-x", "+y", "diff --git a/b.ts b/b.ts", "--- a/b.ts", "+++ b/b.ts", "@@ -5,1 +5,2 @@", " keep", "+add"].join("\r\n")
    const files = parsePatchFiles(patch)
    expect(files.map(({ oldPath, newPath, lines }) => [oldPath, newPath, lines.map((line) => line.kind)])).toEqual([
      ["a/a.ts", "b/a.ts", ["hunk", "removed", "added"]],
      ["a/b.ts", "b/b.ts", ["hunk", "context", "added"]],
    ])
    expect(files[1].lines[2]).toEqual({ kind: "added", text: "add", oldNumber: null, newNumber: 6 })
  })

  it("reads a removed line starting with `-- ` as a line, a bare hunk list as one pathless file", () => {
    const [file] = parsePatchFiles("@@ -1,2 +1,1 @@\n--- not a header\n keep\n")
    expect(file.oldPath).toBeUndefined()
    expect(file.lines.map(({ kind, text }) => [kind, text])).toEqual([["hunk", "@@ -1,2 +1,1 @@"], ["removed", "-- not a header"], ["context", "keep"]])
  })

  it("finds no file in text without a hunk", () => {
    expect(parsePatchFiles("plain text output")).toEqual([])
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
