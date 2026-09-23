import { describe, expect, it } from "vitest"
import { notebookDiff, parseNotebook, serializeNotebook } from "@/lib/notebook"

describe("parseNotebook", () => {
  it("parses a file with no markers as one implicit code cell", () => {
    const cells = parseNotebook("print('hi')\nprint('bye')\n")
    expect(cells).toHaveLength(1)
    expect(cells[0].kind).toBe("code")
    expect(cells[0].source).toBe("print('hi')\nprint('bye')")
  })

  it("splits # %% markers into cell kinds and sources", () => {
    const text = "# %%\na = 1\n\n# %%\nb = 2\n"
    const cells = parseNotebook(text)
    expect(cells.map((cell) => cell.kind)).toEqual(["code", "code"])
    expect(cells.map((cell) => cell.source)).toEqual(["a = 1", "b = 2"])
  })

  it("strips the # prefix from markdown cell bodies", () => {
    const text = "# %% [markdown]\n# # Title\n#\n# Some *prose*.\n"
    const cells = parseNotebook(text)
    expect(cells).toHaveLength(1)
    expect(cells[0].kind).toBe("markdown")
    expect(cells[0].source).toBe("# Title\n\nSome *prose*.")
  })

  it("treats non-blank content before the first marker as a leading code cell", () => {
    const cells = parseNotebook("import os\n\n# %%\nx = 1\n")
    expect(cells.map((cell) => cell.kind)).toEqual(["code", "code"])
    expect(cells[0].source).toBe("import os")
    expect(cells[1].source).toBe("x = 1")
  })
})

describe("serializeNotebook", () => {
  it("round trips a mixed notebook exactly", () => {
    const text = [
      "# %%",
      "a = 1",
      "",
      "# %% [markdown]",
      "# # Title",
      "#",
      "# Body line.",
      "",
      "# %%",
      "b = 2",
      "",
    ].join("\n")
    expect(serializeNotebook(parseNotebook(text))).toBe(text)
  })

  it("normalizes a markerless file to an explicit single code cell", () => {
    expect(serializeNotebook(parseNotebook("print('hi')\nprint('bye')\n"))).toBe(
      "# %%\nprint('hi')\nprint('bye')\n",
    )
  })

  it("restores # prefixes on serialized markdown bodies", () => {
    const cells = parseNotebook("# %% [markdown]\n# # Title\n#\n# Body.\n")
    expect(serializeNotebook(cells)).toBe("# %% [markdown]\n# # Title\n#\n# Body.\n")
  })
})

describe("notebookDiff", () => {
  it("reports zero changes for identical text", () => {
    const text = "# %%\na = 1\n"
    expect(notebookDiff(text, text)).toEqual({ added: 0, modified: 0, removed: 0 })
  })

  it("counts appended and removed cells", () => {
    const before = "# %%\na = 1\n"
    const after = "# %%\na = 1\n\n# %%\nb = 2\n"
    expect(notebookDiff(before, after)).toEqual({ added: 1, modified: 0, removed: 0 })
    expect(notebookDiff(after, before)).toEqual({ added: 0, modified: 0, removed: 1 })
  })

  it("counts an edited cell as modified, not added plus removed", () => {
    const before = "# %%\na = 1\n\n# %%\nb = 2\n"
    const after = "# %%\na = 1\n\n# %%\nb = 3\n"
    expect(notebookDiff(before, after)).toEqual({ added: 0, modified: 1, removed: 0 })
  })

  it("pairs leftover cells of the same kind in order, counting the excess as removed", () => {
    const before = ["# %%", "a = 1", "", "# %%", "b = 2", "", "# %%", "c = 3", ""].join("\n")
    const after = ["# %%", "a = 1", "", "# %%", "b = 99", "", "# %%", "d = 4", ""].join("\n")
    expect(notebookDiff(before, after)).toEqual({ added: 0, modified: 2, removed: 0 })
  })

  it("reports removal when a kind has no leftover counterpart to pair with", () => {
    const before = "# %% [markdown]\n# Notes\n\n# %%\nx = 1\n"
    const after = "# %% [markdown]\n# Notes\n"
    expect(notebookDiff(before, after)).toEqual({ added: 0, modified: 0, removed: 1 })
  })

  it("pairs edits by cell kind", () => {
    const before = "# %% [markdown]\n# Hello\n"
    const after = "# %% [markdown]\n# Goodbye\n"
    expect(notebookDiff(before, after)).toEqual({ added: 0, modified: 1, removed: 0 })
  })

  it("reports a markerless file edit as one modified cell", () => {
    expect(notebookDiff("x = 1\n", "x = 2\n")).toEqual({ added: 0, modified: 1, removed: 0 })
  })

  it("keeps an unchanged middle cell from becoming a modification", () => {
    const before = "# %%\na = 1\n\n# %%\nb = 2\n"
    const after = "# %%\nz = 0\n\n# %%\nb = 2\n"
    expect(notebookDiff(before, after)).toEqual({ added: 0, modified: 1, removed: 0 })
  })
})