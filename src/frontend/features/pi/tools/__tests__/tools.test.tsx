import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { ToolCall, ToolRun } from "../../types"
import { getResultDiff } from "../DiffView"
import { parsePiDisplayDiff, parseUnifiedPatch } from "../patch"
import { ToolCallRow } from "../ToolCallRow"

const editCall: ToolCall = { type: "toolCall", id: "call-1", name: "edit", arguments: { path: "src/app.ts", edits: [{ oldText: "a", newText: "b" }] } }

describe("patch parsing", () => {
  it("pairs removed and added lines of a unified patch", () => {
    const files = parseUnifiedPatch("--- a.ts\n+++ a.ts\n@@ -1,2 +1,2 @@\n keep\n-old\n+new\n")
    expect(files).toHaveLength(1)
    expect(files?.[0].rows.filter((row) => row.type === "line")).toEqual([
      { type: "line", left: { lineNo: 1, text: "keep", type: "context" }, right: { lineNo: 1, text: "keep", type: "context" } },
      { type: "line", left: { lineNo: 2, text: "old", type: "removed" }, right: { lineNo: 2, text: "new", type: "added" } },
    ])
  })

  it("splits a multi-file patch with its paths and drops text that is no patch", () => {
    const files = parseUnifiedPatch("--- a/x.ts\n+++ b/x.ts\n@@ -1 +1 @@\n-a\n+b\n--- a/y.ts\n+++ b/y.ts\n@@ -3 +3 @@\n-c\n+d\n\\ No newline at end of file")
    expect(files?.map(({ oldPath, newPath }) => [oldPath, newPath])).toEqual([["a/x.ts", "b/x.ts"], ["a/y.ts", "b/y.ts"]])
    expect(files?.[1].rows.at(-1)).toEqual({ type: "hunk", text: "\\ No newline at end of file" })
    expect(parseUnifiedPatch("plain text output")).toBeNull()
  })

  it("parses Pi's display diff and derives new line numbers of context lines", () => {
    const files = parsePiDisplayDiff("   ...\n  4 keep\n- 5 old\n+ 5 new\n+ 6 more\n  6 tail")
    const lines = files?.[0].rows.filter((row) => row.type === "line") ?? []
    expect(lines.at(-1)).toEqual({ type: "line", left: { lineNo: 6, text: "tail", type: "context" }, right: { lineNo: 7, text: "tail", type: "context" } })
    expect(parsePiDisplayDiff("plain text output")).toBeNull()
  })

  it("prefers the unified patch of a Pi edit result", () => {
    expect(getResultDiff({ diff: "+1 x", patch: "--- a\n+++ a" })).toBe("--- a\n+++ a")
    expect(getResultDiff({ diff: "+1 x" })).toBe("+1 x")
    expect(getResultDiff(undefined)).toBeNull()
  })
})

describe("ToolCallRow", () => {
  it("shows a finished edit as a paired diff, open without a click", () => {
    const run: ToolRun = { running: false, isError: false, durationMs: 1200, result: { content: [{ type: "text", text: "Successfully replaced 1 block(s)" }], details: { diff: "- 1 old\n+ 1 new" } } }
    render(<ToolCallRow call={editCall} run={run} />)
    expect(screen.getByText("1.2s")).toBeTruthy()
    expect(screen.getByRole("button", { name: /done edit/ }).getAttribute("aria-expanded")).toBe("true")
    expect(screen.getByText("old")).toBeTruthy()
    expect(screen.getByText("new")).toBeTruthy()
    expect(screen.queryByText("Successfully replaced 1 block(s)")).toBeNull()
  })

  it("opens the file path without toggling the row", () => {
    const onOpenFile = vi.fn()
    render(<ToolCallRow call={editCall} onOpenFile={onOpenFile} />)
    fireEvent.click(screen.getByRole("button", { name: "src/app.ts" }))
    expect(onOpenFile).toHaveBeenCalledWith("src/app.ts")
    expect(screen.getByRole("button", { name: /not run edit/ }).getAttribute("aria-expanded")).toBe("true")
  })
})
