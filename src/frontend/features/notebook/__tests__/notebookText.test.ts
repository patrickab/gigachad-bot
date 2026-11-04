import { afterEach, describe, expect, it, vi } from "vitest"
import type { JsonObject, NotebookCellContent, NotebookContent } from "../notebookSync"
import { applyNotebookText, cellAtLine, notebookToText, parseNotebookText } from "../notebookText"
import { createNotebookTextBridge, NOTEBOOK_TEXT_DELAY_MS } from "../useNotebookTextBridge"
import { FakeNotebook } from "./fakeNotebook"

const stream = (text: string): JsonObject => ({ output_type: "stream", name: "stdout", text })
const code = (id: string, source: string, output?: string): NotebookCellContent => ({
  cell_type: "code", id, source, metadata: { tag: id }, outputs: output ? [stream(output)] : [], execution_count: output ? 1 : null,
})
const markdown = (id: string, source: string): NotebookCellContent => ({ cell_type: "markdown", id, source, metadata: {} })
const raw = (id: string, source: string): NotebookCellContent => ({ cell_type: "raw", id, source, metadata: {} })
const notebook = (cells: NotebookCellContent[], minor = 5): NotebookContent => ({ cells, metadata: {}, nbformat: 4, nbformat_minor: minor })

describe("percent text", () => {
  it("writes every kind of cell", () => {
    const text = notebookToText(notebook([code("a", "import numpy as np"), markdown("b", "# Title\n\nSome *text*"), raw("c", "raw text"), code("d", "")]))
    expect(text).toBe("# %%\nimport numpy as np\n\n# %% [markdown]\n# # Title\n#\n# Some *text*\n\n# %% [raw]\n# raw text\n\n# %%\n")
  })

  it("round trips cells with empty lines", () => {
    const cells = [code("a", "x = 1\n\ny = 2"), markdown("b", "one\n\n  two"), raw("c", ""), code("d", "")]
    expect(parseNotebookText(notebookToText(notebook(cells)))).toEqual(cells.map(({ cell_type, source }) => ({ cell_type, source })))
  })

  it("is empty for an empty notebook", () => {
    expect(notebookToText(notebook([]))).toBe("")
    expect(parseNotebookText("")).toEqual([])
    expect(parseNotebookText("\n\n")).toEqual([])
  })

  it("makes text before the first marker a code cell, titles and [md] aside", () => {
    expect(parseNotebookText("import os\n\n# %% Load the data\nx\n# %% [md] Notes\n#text\nplain\n")).toEqual([
      { cell_type: "code", source: "import os" },
      { cell_type: "code", source: "x" },
      { cell_type: "markdown", source: "text\nplain" },
    ])
  })

  it("finds the cell of a line", () => {
    const text = "# %%\na\n\n# %% [markdown]\n# b\n"
    expect([1, 2, 3, 4, 5, 9].map((line) => cellAtLine(text, line))).toEqual([0, 0, 0, 1, 1, 1])
    expect(cellAtLine("lead\n# %%\nx\n", 1)).toBe(0)
    expect(cellAtLine("lead\n# %%\nx\n", 3)).toBe(1)
    expect(cellAtLine("\n# %%\nx\n", 1)).toBe(0)
    expect(cellAtLine("", 1)).toBeNull()
  })

  it("keeps a markdown or raw line starting with %% inside its cell", () => {
    const cells = [markdown("m", "```mermaid\n%% note\ngraph\n```\n%%"), raw("r", "%%")]
    const text = notebookToText(notebook(cells))
    expect(text).toBe("# %% [markdown]\n# ```mermaid\n#%% note\n# graph\n# ```\n#%%\n\n# %% [raw]\n#%%\n")
    expect(parseNotebookText(text)).toEqual(cells.map(({ cell_type, source }) => ({ cell_type, source })))
  })
})

describe("applying the text to the model", () => {
  const cells = () => [code("a", "a = 1", "A"), markdown("m", "# Doc"), code("b", "b = 2", "B"), code("c", "c = 3", "C")]

  for (const minor of [5, 4]) {
    describe(`nbformat 4.${minor}`, () => {
      const edited = (edit: (text: string) => string) => {
        const model = new FakeNotebook(notebook(cells(), minor))
        applyNotebookText(model, edit(notebookToText(model.toJSON())))
        return model
      }
      const outputsOf = (model: FakeNotebook) => model.cells.map((cell) => (cell.cell_type === "code" ? cell.outputs.map((output) => output.text).join("") : "-"))

      it("keeps the outputs of an edited cell", () => {
        const model = edited((text) => text.replace("b = 2", "b = 20"))
        expect(model.cells.map((cell) => cell.source)).toEqual(["a = 1", "# Doc", "b = 20", "c = 3"])
        expect(outputsOf(model)).toEqual(["A", "-", "B", "C"])
        expect(model.cells[2].metadata).toEqual({ tag: "b" })
        expect(model.cells[2].execution_count).toBe(1)
      })

      it("keeps the neighbours' outputs around an inserted cell", () => {
        const model = edited((text) => text.replace("# %% [markdown]", "# %%\nnew = 0\n\n# %% [markdown]"))
        expect(model.cells.map((cell) => cell.source)).toEqual(["a = 1", "new = 0", "# Doc", "b = 2", "c = 3"])
        expect(outputsOf(model)).toEqual(["A", "", "-", "B", "C"])
        expect(model.cells[1].execution_count).toBeNull()
      })

      it("deletes a cell", () => {
        const model = edited((text) => text.replace("# %%\nb = 2\n\n", ""))
        expect(model.cells.map((cell) => cell.source)).toEqual(["a = 1", "# Doc", "c = 3"])
        expect(outputsOf(model)).toEqual(["A", "-", "C"])
      })

      it("tells an edited cell from an inserted one in the same gap", () => {
        const model = edited((text) => text.replace("# %%\nb = 2", "# %%\nfresh()\n\n# %%\nb = 2\nprint(b)"))
        expect(model.cells.map((cell) => cell.source)).toEqual(["a = 1", "# Doc", "fresh()", "b = 2\nprint(b)", "c = 3"])
        expect(outputsOf(model)).toEqual(["A", "-", "", "B", "C"])
      })

      it("replaces a cell whose type changed", () => {
        const model = edited((text) => text.replace("# %%\nb = 2", "# %% [markdown]\n# b = 2"))
        expect(model.cells.map((cell) => cell.cell_type)).toEqual(["code", "markdown", "markdown", "code"])
        expect(model.cells[2].source).toBe("b = 2")
        expect(outputsOf(model)).toEqual(["A", "-", "-", "C"])
      })

      it("keeps a markdown cell with a %% line one cell on a save", () => {
        const model = new FakeNotebook(notebook([markdown("m", "a\n%% mermaid comment\n%%"), code("a", "a = 1", "A")], minor))
        applyNotebookText(model, notebookToText(model.toJSON()).replace("a = 1", "a = 2"))
        expect(model.cells.map((cell) => [cell.cell_type, cell.source])).toEqual([["markdown", "a\n%% mermaid comment\n%%"], ["code", "a = 2"]])
      })

      it("keeps the exact source of cells the text did not change", () => {
        const model = new FakeNotebook(notebook([code("x", "x\n", "X"), code("y", "y = 1\n\n\n"), code("a", "a = 1")], minor))
        const text = notebookToText(model.toJSON())
        applyNotebookText(model, text)
        expect(model.cells.map((cell) => cell.source)).toEqual(["x\n", "y = 1\n\n\n", "a = 1"])
        applyNotebookText(model, text.replace("a = 1", "a = 2"))
        expect(model.cells.map((cell) => cell.source)).toEqual(["x\n", "y = 1\n\n\n", "a = 2"])
        expect(model.cells[0].updates).toEqual([])
        expect(model.cells[1].updates).toEqual([])
      })
    })
  }
})

describe("text bridge", () => {
  afterEach(() => { vi.useRealTimers() })

  it("saves, persists, and reports only changes it did not hand out", async () => {
    vi.useFakeTimers()
    const model = new FakeNotebook(notebook([code("a", "a = 1", "A")]))
    const persist = vi.fn(async () => true)
    const run = vi.fn()
    const bridge = createNotebookTextBridge(model, persist, run)
    const heard: string[] = []
    bridge.subscribe((text) => heard.push(text))
    expect(bridge.extension).toBe(".py")

    const saved = await bridge.save(`${bridge.text()}\n# %%\nb = 2\n`)
    expect(saved).toBe("# %%\na = 1\n\n# %%\nb = 2\n")
    expect(persist).toHaveBeenCalledTimes(1)
    expect(model.cells[0].outputs).toEqual([stream("A")])
    vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toEqual([])

    model.edit(0, "a = 3")
    vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toEqual(["# %%\na = 3\n\n# %%\nb = 2\n"])

    bridge.runAt(5)
    expect(run).toHaveBeenCalledWith(1)
  })

  it("rejects when persisting fails", async () => {
    const bridge = createNotebookTextBridge(new FakeNotebook(notebook([])), async () => false, () => {})
    await expect(bridge.save(`${bridge.text()}# %%\nx\n`)).rejects.toThrow("Saving the notebook failed")
  })

  it("rejects a save while the notebook has changes the text has not, and hands the new text over at once", async () => {
    vi.useFakeTimers()
    const model = new FakeNotebook(notebook([code("a", "a = 1"), code("b", "b = 2")]))
    const persist = vi.fn(async () => true)
    const bridge = createNotebookTextBridge(model, persist, () => {})
    const heard: string[] = []
    bridge.subscribe((text) => heard.push(text))
    const text = bridge.text()

    model.edit(1, "b = 3")
    await expect(bridge.save(text.replace("a = 1", "a = 10"))).rejects.toThrow("The notebook changed, write again")
    expect(heard).toEqual(["# %%\na = 1\n\n# %%\nb = 3\n"])
    expect(model.cells.map((cell) => cell.source)).toEqual(["a = 1", "b = 3"])
    expect(persist).not.toHaveBeenCalled()
    vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toHaveLength(1)

    await expect(bridge.save(heard[0].replace("a = 1", "a = 10"))).resolves.toBe("# %%\na = 10\n\n# %%\nb = 3\n")
  })

  it("never reports the text being saved while the persist runs", async () => {
    vi.useFakeTimers()
    const model = new FakeNotebook(notebook([code("a", "a = 1")]))
    const persisted = Promise.withResolvers<boolean>()
    const bridge = createNotebookTextBridge(model, () => persisted.promise, () => {})
    const heard: string[] = []
    bridge.subscribe((text) => heard.push(text))
    const saving = bridge.save(bridge.text().replace("a = 1", "a = 2"))
    vi.advanceTimersByTime(2 * NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toEqual([])
    persisted.resolve(true)
    await expect(saving).resolves.toBe("# %%\na = 2\n")
    expect(heard).toEqual([])
  })

  it("falls back to .py for an extension the backend refuses", () => {
    const extensionOf = (file_extension: JsonObject[string]) =>
      createNotebookTextBridge(new FakeNotebook({ ...notebook([]), metadata: { language_info: { file_extension } } }), async () => true, () => {}).extension
    expect(extensionOf(".jl")).toBe(".jl")
    for (const invalid of ["", "py", ".c++", ".toolongext", 3]) expect(extensionOf(invalid)).toBe(".py")
  })
})
