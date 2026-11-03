import { describe, expect, it } from "vitest"
import {
  applyIncomingNotebook,
  diffNotebook,
  notebookOf,
  normalizeNotebook,
  parseNotebook,
  rejoinNotebook,
  serializeNotebook,
  splitLines,
  type JsonObject,
  type NotebookCellContent,
  type NotebookContent,
} from "../notebookSync"
import { FakeNotebook } from "./fakeNotebook"

const code = (id: string, source: string, extra: Partial<NotebookCellContent> = {}): NotebookCellContent => ({
  cell_type: "code", id, source, metadata: {}, outputs: [], execution_count: null, ...extra,
})
const markdown = (id: string, source: string, extra: Partial<NotebookCellContent> = {}): NotebookCellContent => ({ cell_type: "markdown", id, source, metadata: {}, ...extra })
const notebook = (cells: NotebookCellContent[], metadata: JsonObject = {}): NotebookContent => ({ cells, metadata, nbformat: 4, nbformat_minor: 5 })
const stream = (text: string): JsonObject => ({ output_type: "stream", name: "stdout", text })

/** The notebook the editor ends with after `incoming` was merged into `current`, which started as `base`. */
function merged(base: NotebookContent, current: NotebookContent, incoming: NotebookContent): NotebookContent {
  const model = new FakeNotebook(current)
  applyIncomingNotebook(model, base, incoming)
  return model.toJSON()
}
const sourcesOf = (content: NotebookContent) => content.cells.map((cell) => cell.source)
const idsOf = (content: NotebookContent) => content.cells.map((cell) => cell.id)

describe("serialization round trip", () => {
  const fixture = notebook([
    markdown("m1", "# Title\n\nText with ünïcode and 日本語 😀\n"),
    code("c1", "x = 1\nprint('é')", {
      execution_count: 3,
      metadata: { tags: ["a"] },
      outputs: [
        stream("one\ntwo\n"),
        { output_type: "execute_result", execution_count: 3, metadata: {}, data: { "text/plain": "1\n2", "image/png": "iVBOR\n", "application/json": { b: 1, a: [] }, "text/html": "" } },
        { output_type: "error", ename: "E", evalue: "v", traceback: ["a", "b\n"] },
      ],
    }),
    code("c2", ""),
  ], { kernelspec: { name: "python3", display_name: "Python 3", language: "python" }, language_info: { name: "python", version: "3.12" } })
  const fixtureText = serializeNotebook(fixture)

  it("writes Jupyter's style", () => {
    expect(fixtureText.endsWith("}\n")).toBe(true)
    expect(fixtureText).toContain('\n  {\n   "cell_type": "markdown",')
    expect(fixtureText).toContain('"source": [\n    "# Title\\n",\n    "\\n",')
    expect(fixtureText).toContain('"source": []')
    expect(fixtureText).toContain('"text/plain": [\n       "1\\n",\n       "2"\n      ]')
    // JSON mimes and base64 stay values
    expect(fixtureText).toContain('"image/png": "iVBOR\\n"')
    expect(fixtureText).toContain('"application/json": {\n       "a": [],\n       "b": 1\n      }')
    // keys sorted at every level
    expect(Object.keys(JSON.parse(fixtureText))).toEqual(["cells", "metadata", "nbformat", "nbformat_minor"])
  })

  it("round trips a notebook from text and from the editor's strings", () => {
    const original = parseNotebook(fixtureText)
    expect(serializeNotebook(original, original)).toBe(fixtureText)
    expect(serializeNotebook(rejoinNotebook(original), original)).toBe(fixtureText)
    expect(serializeNotebook(rejoinNotebook(original))).toBe(fixtureText)
  })

  it("splits lines like Python's splitlines", () => {
    expect(splitLines("")).toEqual([])
    expect(splitLines("a\nb")).toEqual(["a\n", "b"])
    expect(splitLines("a\r\nb\rc\u2028d\x0be\n")).toEqual(["a\r\n", "b\r", "c\u2028", "d\x0b", "e\n"])
    expect(splitLines("\n\n")).toEqual(["\n", "\n"])
  })

  it("keeps a stored string where the file had one and a stored list where the text is unchanged", () => {
    const stored = parseNotebook(JSON.stringify({ ...fixture, cells: [{ ...markdown("m1", "a\nb"), source: "a\nb" }, { ...markdown("m2", "x\ny"), source: ["x\n", "y"] }] }, null, 1))
    const edited = rejoinNotebook(stored)
    edited.cells[0].source = "a\nb\nc"
    const written = JSON.parse(serializeNotebook(edited, stored)) as NotebookContent
    expect(written.cells[0].source).toBe("a\nb\nc")
    expect(written.cells[1].source).toEqual(["x\n", "y"])
    edited.cells[1].source = "x\ny\nz"
    expect((JSON.parse(serializeNotebook(edited, stored)) as NotebookContent).cells[1].source).toEqual(["x\n", "y\n", "z"])
  })

  it("keeps the key order of a file that was not sorted", () => {
    const text = '{\n "nbformat": 4,\n "nbformat_minor": 5,\n "metadata": {},\n "cells": [\n  {\n   "id": "a",\n   "cell_type": "markdown",\n   "source": [],\n   "metadata": {}\n  }\n ]\n}\n'
    const original = parseNotebook(text)
    expect(serializeNotebook(rejoinNotebook(original), original)).toBe(text)
  })

  it("parses a blank file as an empty notebook and refuses what is not a notebook", () => {
    expect(parseNotebook("  \n")).toEqual({ cells: [], metadata: {}, nbformat: 4, nbformat_minor: 5 })
    expect(() => parseNotebook("{}")).toThrow()
    expect(() => parseNotebook("[1]")).toThrow()
    expect(() => parseNotebook("{")).toThrow()
  })
})

describe("what the library adds is not saved", () => {
  it("drops cell metadata.editable and trusted unless the stored cell had them", () => {
    const original = notebook([code("a", "1", { metadata: { editable: false, scrolled: true } }), code("b", "2")])
    const editor = notebook([code("a", "1", { metadata: { editable: false, scrolled: true, trusted: true } }), code("b", "2", { metadata: { editable: true, trusted: true } })])
    const written = JSON.parse(serializeNotebook(editor, original)) as NotebookContent
    expect(written.cells.map((cell) => cell.metadata)).toEqual([{ editable: false, scrolled: true }, {}])
  })

  it("keeps orig_nbformat of the stored notebook", () => {
    const original = notebook([], { orig_nbformat: 4, orig_nbformat_minor: 1 })
    expect(JSON.parse(serializeNotebook(notebook([]), original)).metadata).toEqual({ orig_nbformat: 4, orig_nbformat_minor: 1 })
  })

  it("writes no ids into a notebook that has none", () => {
    const legacy: NotebookContent = { ...notebook([{ ...code("x", "1"), id: undefined }]), nbformat_minor: 4 }
    delete legacy.cells[0].id
    const editor: NotebookContent = { ...legacy, cells: [code("generated-0", "1"), code("generated-1", "2")] }
    for (const cell of JSON.parse(serializeNotebook(editor, legacy)).cells) expect(cell.id).toBeUndefined()
    expect(JSON.parse(serializeNotebook({ ...editor, nbformat_minor: 5 })).cells[0].id).toBe("generated-0")
  })

  it("drops the transient part of outputs", () => {
    const editor = notebook([code("a", "1", { outputs: [{ output_type: "display_data", data: {}, metadata: {}, transient: { display_id: "d" } }] })])
    expect(JSON.parse(serializeNotebook(editor)).cells[0].outputs[0].transient).toBeUndefined()
  })

  it("compares notebooks by data, not by style", () => {
    const listed = notebook([markdown("a", "x\ny", { source: ["x\n", "y"] })], { b: 1, a: 2 })
    const joined = notebook([markdown("a", "x\ny", { metadata: { editable: true } })], { a: 2, b: 1 })
    expect(normalizeNotebook(listed, listed)).toBe(normalizeNotebook(joined, listed))
    expect(normalizeNotebook(listed)).not.toBe(normalizeNotebook(notebook([markdown("a", "x\nz")], { a: 2, b: 1 })))
  })
})

describe("diff policy", () => {
  const base = notebook([code("a", "a = 1"), code("b", "b = 1"), markdown("c", "# c")], { language: "py", note: "n" })

  it("is empty when the file did not change", () => {
    expect(diffNotebook(base, base, base)).toEqual([])
    const edited = notebook([code("a", "a = 2"), code("b", "b = 1"), markdown("c", "# c")], { language: "py", note: "n" })
    expect(diffNotebook(base, edited, base)).toEqual([])
    expect(merged(base, edited, base)).toEqual(edited)
  })

  it("is empty when the editor already shows the file", () => {
    const incoming = notebook([code("a", "a = 2"), code("b", "b = 1"), markdown("c", "# c")], { language: "py", note: "n" })
    expect(diffNotebook(base, incoming, incoming)).toEqual([])
  })

  describe("source", () => {
    const incoming = notebook([code("a", "a = 2"), code("b", "b = 1"), markdown("c", "# c")], base.metadata)

    it("takes the file's text when only the file changed it", () => {
      expect(sourcesOf(merged(base, base, incoming))).toEqual(["a = 2", "b = 1", "# c"])
    })

    it("takes the file's text when both changed it", () => {
      const local = notebook([code("a", "a = 99"), code("b", "b = 1"), markdown("c", "# c")], base.metadata)
      expect(sourcesOf(merged(base, local, incoming))[0]).toBe("a = 2")
    })

    it("keeps the editor's text when only the editor changed it", () => {
      const local = notebook([code("a", "a = 1"), code("b", "b = 5"), markdown("c", "# c")], base.metadata)
      expect(sourcesOf(merged(base, local, incoming))).toEqual(["a = 2", "b = 5", "# c"])
    })

    it("touches only the text that differs", () => {
      const model = new FakeNotebook(notebook([code("a", "import numpy as np\nx = 1\nprint(x)\n")]))
      applyIncomingNotebook(model, notebook([code("a", "import numpy as np\nx = 1\nprint(x)\n")]), notebook([code("a", "import numpy as np\nx = 22\nprint(x)\n")]))
      expect(model.cells[0].source).toBe("import numpy as np\nx = 22\nprint(x)\n")
      expect(model.cells[0].updates).toEqual([{ start: 23, end: 24, value: "22" }])
    })

    it("never cuts a surrogate pair", () => {
      const model = new FakeNotebook(notebook([code("a", "x😀y")]))
      applyIncomingNotebook(model, notebook([code("a", "x😀y")]), notebook([code("a", "x😁y")]))
      expect(model.cells[0].source).toBe("x😁y")
      expect(model.cells[0].updates).toEqual([{ start: 1, end: 3, value: "😁" }])
    })

    it("compares the file's lists of lines with the editor's strings", () => {
      const listed = notebook([code("a", "a = 1", { source: ["a = 1"] }), code("b", "b = 1"), markdown("c", "# c")], base.metadata)
      expect(diffNotebook(base, base, listed)).toEqual([])
    })
  })

  describe("outputs and execution count", () => {
    const withOutput = (text: string, count: number | null) => notebook([code("a", "a = 1", { outputs: [stream(text)], execution_count: count }), code("b", "b = 1"), markdown("c", "# c")], base.metadata)

    it("takes the file's outputs and count when only the file changed them", () => {
      const result = merged(base, base, withOutput("hi\n", 4))
      expect(result.cells[0].outputs).toEqual([stream("hi\n")])
      expect(result.cells[0].execution_count).toBe(4)
    })

    it("takes the file's when both changed them", () => {
      const result = merged(base, withOutput("local\n", 9), withOutput("file\n", 4))
      expect(result.cells[0].outputs).toEqual([stream("file\n")])
      expect(result.cells[0].execution_count).toBe(4)
    })

    it("keeps the editor's when only the editor changed them", () => {
      const result = merged(withOutput("old\n", 1), withOutput("running\n", 2), notebook([code("a", "a = 3", { outputs: [stream("old\n")], execution_count: 1 }), code("b", "b = 1"), markdown("c", "# c")], base.metadata))
      expect(result.cells[0].outputs).toEqual([stream("running\n")])
      expect(result.cells[0].execution_count).toBe(2)
      expect(result.cells[0].source).toBe("a = 3")
    })

    it("clears outputs the file cleared", () => {
      const result = merged(withOutput("old\n", 1), withOutput("old\n", 1), base)
      expect(result.cells[0].outputs).toEqual([])
      expect(result.cells[0].execution_count).toBeNull()
    })
  })

  describe("metadata", () => {
    it("merges cell metadata key by key", () => {
      const start = notebook([code("a", "1", { metadata: { tags: ["x"], keep: 1, drop: true } })])
      const local = notebook([code("a", "1", { metadata: { tags: ["x"], keep: 2, drop: true } })])
      const file = notebook([code("a", "1", { metadata: { tags: ["y"], keep: 1, drop: null, added: "n" } })])
      delete (file.cells[0].metadata as JsonObject).drop
      const result = merged(start, local, file)
      expect(result.cells[0].metadata).toEqual({ tags: ["y"], keep: 2, added: "n" })
    })

    it("lets the file's value win a key both changed", () => {
      const start = notebook([code("a", "1", { metadata: { k: 1 } })])
      expect(merged(start, notebook([code("a", "1", { metadata: { k: 2 } })]), notebook([code("a", "1", { metadata: { k: 3 } })])).cells[0].metadata).toEqual({ k: 3 })
    })

    it("ignores what the library adds", () => {
      const editor = new FakeNotebook(base, { injectEditable: true })
      const fileEdit = notebook([code("a", "a = 2"), code("b", "b = 1"), markdown("c", "# c")], base.metadata)
      applyIncomingNotebook(editor, base, fileEdit)
      expect(editor.cells[0].metadata).toEqual({ editable: true })
      expect(editor.cells[0].source).toBe("a = 2")
      expect(diffNotebook(base, editor.toJSON(), base)).toEqual([])
    })

    it("merges notebook metadata key by key", () => {
      const file = notebook(base.cells, { language: "r", note: "n", fresh: { a: 1 } })
      const local = notebook(base.cells, { language: "py", note: "mine" })
      const model = new FakeNotebook(local)
      applyIncomingNotebook(model, base, file)
      expect(model.metadata).toEqual({ language: "r", note: "mine", fresh: { a: 1 } })
    })

    it("removes a notebook metadata key the file removed", () => {
      const model = new FakeNotebook(base)
      applyIncomingNotebook(model, base, notebook(base.cells, { language: "py" }))
      expect(model.metadata).toEqual({ language: "py" })
    })

    it("leaves orig_nbformat alone", () => {
      expect(diffNotebook(base, base, notebook(base.cells, { ...base.metadata, orig_nbformat: 4 }))).toEqual([])
    })
  })

  describe("cells", () => {
    it("inserts a cell the file added", () => {
      const incoming = notebook([base.cells[0], code("n", "new"), base.cells[1], base.cells[2]], base.metadata)
      const result = merged(base, base, incoming)
      expect(idsOf(result)).toEqual(["a", "n", "b", "c"])
      expect(result.cells[1]).toEqual(code("n", "new"))
    })

    it("inserts a cell the file added at the start and the end", () => {
      const incoming = notebook([code("s", "start"), ...base.cells, code("e", "end")], base.metadata)
      expect(idsOf(merged(base, base, incoming))).toEqual(["s", "a", "b", "c", "e"])
    })

    it("deletes a cell the file deleted, edited here or not", () => {
      const incoming = notebook([base.cells[0], base.cells[2]], base.metadata)
      expect(idsOf(merged(base, base, incoming))).toEqual(["a", "c"])
      const local = notebook([base.cells[0], code("b", "b = edited"), base.cells[2]], base.metadata)
      expect(idsOf(merged(base, local, incoming))).toEqual(["a", "c"])
    })

    it("keeps a cell only the editor added, where it is", () => {
      const local = notebook([base.cells[0], code("mine", "mine"), base.cells[1], base.cells[2]], base.metadata)
      const incoming = notebook([code("n", "new"), ...base.cells], base.metadata)
      expect(idsOf(merged(base, local, incoming))).toEqual(["n", "a", "mine", "b", "c"])
    })

    it("keeps a cell only the editor deleted deleted, unless the file changed it", () => {
      const local = notebook([base.cells[0], base.cells[2]], base.metadata)
      const untouched = notebook([base.cells[0], code("b", "b = 1"), base.cells[2], code("n", "n")], base.metadata)
      expect(idsOf(merged(base, local, untouched))).toEqual(["a", "c", "n"])
      const touched = notebook([base.cells[0], code("b", "b = file"), base.cells[2]], base.metadata)
      const result = merged(base, local, touched)
      expect(idsOf(result)).toEqual(["a", "b", "c"])
      expect(result.cells[1].source).toBe("b = file")
    })

    it("moves cells the way the file moved them", () => {
      const incoming = notebook([base.cells[2], base.cells[0], base.cells[1]], base.metadata)
      expect(idsOf(merged(base, base, incoming))).toEqual(["c", "a", "b"])
      const reversed = notebook([base.cells[2], base.cells[1], base.cells[0]], base.metadata)
      expect(idsOf(merged(base, base, reversed))).toEqual(["c", "b", "a"])
    })

    it("lets the file's order win when both moved cells", () => {
      const local = notebook([base.cells[1], base.cells[0], base.cells[2]], base.metadata)
      const incoming = notebook([base.cells[2], base.cells[0], base.cells[1]], base.metadata)
      expect(idsOf(merged(base, local, incoming))).toEqual(["c", "a", "b"])
    })

    it("keeps the editor's order when only the editor moved cells", () => {
      const local = notebook([base.cells[1], base.cells[0], base.cells[2]], base.metadata)
      const incoming = notebook([base.cells[0], base.cells[1], base.cells[2], code("n", "n")], base.metadata)
      expect(idsOf(merged(base, local, incoming))).toEqual(["b", "a", "c", "n"])
    })

    it("keeps the editor's cells added next to a move by the file", () => {
      const local = notebook([base.cells[0], code("mine", "mine"), base.cells[1], base.cells[2]], base.metadata)
      const incoming = notebook([base.cells[2], base.cells[0], base.cells[1]], base.metadata)
      expect(idsOf(merged(base, local, incoming))).toEqual(["c", "a", "mine", "b"])
    })

    it("applies moves, inserts and edits of one change together", () => {
      const incoming = notebook([base.cells[1], code("n", "new"), code("a", "a = edited"), markdown("c", "# c edited")], base.metadata)
      const result = merged(base, base, incoming)
      expect(idsOf(result)).toEqual(["b", "n", "a", "c"])
      expect(sourcesOf(result)).toEqual(["b = 1", "new", "a = edited", "# c edited"])
    })

    it("replaces a cell whose type the file changed", () => {
      const incoming = notebook([base.cells[0], base.cells[1], code("c", "c = 1")], base.metadata)
      const result = merged(base, base, incoming)
      expect(result.cells[2]).toEqual(code("c", "c = 1"))
    })

    it("keeps the type the editor changed", () => {
      const local = notebook([base.cells[0], base.cells[1], code("c", "# c")], base.metadata)
      const incoming = notebook([base.cells[0], base.cells[1], markdown("c", "# c edited")], base.metadata)
      const result = merged(base, local, incoming)
      expect(result.cells[2]).toEqual(code("c", "# c edited"))
    })

    it("applies outside the undo stack", () => {
      const model = new FakeNotebook(base)
      applyIncomingNotebook(model, base, notebook([code("a", "a = 2"), code("n", "n"), base.cells[2]], base.metadata))
      expect(model.transactions.length).toBeGreaterThan(0)
      expect(model.transactions.every((undoable) => undoable === false)).toBe(true)
    })

    it("does not open a transaction when there is nothing to do", () => {
      const model = new FakeNotebook(base)
      applyIncomingNotebook(model, base, base)
      expect(model.transactions).toEqual([])
    })
  })

  describe("notebooks without ids", () => {
    const withoutIds = (cells: [string, string][]): NotebookContent => ({
      cells: cells.map(([type, source]) => ({ cell_type: type, source, metadata: {}, ...(type === "code" ? { outputs: [], execution_count: null } : {}) })),
      metadata: {}, nbformat: 4, nbformat_minor: 4,
    })

    it("matches cells by position, whatever ids the model made", () => {
      const start = withoutIds([["code", "a"], ["code", "b"], ["markdown", "c"]])
      const model = new FakeNotebook(start)
      applyIncomingNotebook(model, start, withoutIds([["code", "a2"], ["code", "b"], ["markdown", "c"], ["code", "d"]]))
      expect(model.cells.map((cell) => cell.source)).toEqual(["a2", "b", "c", "d"])
      expect(model.cells[3].cell_type).toBe("code")
      expect(diffNotebook(start, model.toJSON(), withoutIds([["code", "a2"], ["code", "b"], ["markdown", "c"], ["code", "d"]]))).toEqual([])
    })

    it("keeps the editor's edit of a cell the file did not touch", () => {
      const start = withoutIds([["code", "a"], ["code", "b"]])
      const model = new FakeNotebook(start)
      model.edit(1, "b mine")
      applyIncomingNotebook(model, start, withoutIds([["code", "a2"], ["code", "b"]]))
      expect(model.cells.map((cell) => cell.source)).toEqual(["a2", "b mine"])
    })

    it("deletes the cells the file dropped from the end", () => {
      const start = withoutIds([["code", "a"], ["code", "b"], ["code", "c"]])
      const model = new FakeNotebook(start)
      applyIncomingNotebook(model, start, withoutIds([["code", "a"]]))
      expect(model.cells.map((cell) => cell.source)).toEqual(["a"])
    })

    it("saves no ids, also for a cell the model added with one", () => {
      const start = withoutIds([["code", "a"], ["markdown", "b"]])
      const model = new FakeNotebook(start)
      model.insertCell(1, { cell_type: "code", source: "new", metadata: {}, outputs: [], execution_count: null })
      expect(model.cells[1].id).toBeDefined()
      const saved = JSON.parse(serializeNotebook(notebookOf(model), start)) as NotebookContent
      expect(saved.nbformat_minor).toBe(4)
      expect(sourcesOf(rejoinNotebook(saved))).toEqual(["a", "new", "b"])
      for (const cell of saved.cells) expect(cell).not.toHaveProperty("id")
    })
  })
})
