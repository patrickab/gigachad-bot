import { afterEach, describe, expect, it, vi } from "vitest"
import type { JsonObject, NotebookCellContent, NotebookContent } from "../notebookSync"
import { createNotebookTextStore, NOTEBOOK_TEXT_DELAY_MS } from "../notebookTextStore"
import { FakeNotebook } from "./fakeNotebook"

const stream = (text: string): JsonObject => ({ output_type: "stream", name: "stdout", text })
const code = (id: string, source: string, output?: string): NotebookCellContent => ({
  cell_type: "code", id, source, metadata: {}, outputs: output ? [stream(output)] : [], execution_count: output ? 1 : null,
})
const notebook = (cells: NotebookCellContent[]): NotebookContent => ({ cells, metadata: {}, nbformat: 4, nbformat_minor: 5 })

/** A store bound to `model`, with the texts its follower got. */
function followed(model: FakeNotebook, persist: () => Promise<boolean> = async () => true, run: (index: number) => void = () => {}) {
  const store = createNotebookTextStore()
  store.bind(model, persist, run)
  const heard: string[] = []
  store.follow((text) => heard.push(text))
  return { store, heard }
}

describe("notebook text store", () => {
  afterEach(() => { vi.useRealTimers() })

  it("saves, persists, and hands out only changes Neovim did not make", async () => {
    vi.useFakeTimers()
    const model = new FakeNotebook(notebook([code("a", "a = 1", "A")]))
    const persist = vi.fn(async () => true)
    const run = vi.fn()
    const { store, heard } = followed(model, persist, run)
    expect(store.state()).toEqual({ extension: ".py", text: "# %%\na = 1\n" })

    const saved = await store.save(`${store.state()?.text}\n# %%\nb = 2\n`)
    expect(saved).toBe("# %%\na = 1\n\n# %%\nb = 2\n")
    expect(store.state()?.text).toBe(saved)
    expect(persist).toHaveBeenCalledTimes(1)
    expect(model.cells[0].outputs).toEqual([stream("A")])
    vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toEqual([])

    model.edit(0, "a = 3")
    vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toEqual(["# %%\na = 3\n\n# %%\nb = 2\n"])
    expect(store.state()?.text).toBe(heard[0])

    store.runAt(5)
    expect(run).toHaveBeenCalledWith(1)
  })

  it("rejects when persisting fails", async () => {
    const store = createNotebookTextStore()
    store.bind(new FakeNotebook(notebook([])), async () => false, () => {})
    await expect(store.save("# %%\nx\n")).rejects.toThrow("Saving the notebook failed")
  })

  it("rejects a save while unbound", async () => {
    await expect(createNotebookTextStore().save("# %%\nx\n")).rejects.toThrow("The notebook is not loaded")
  })

  it("rejects a save while the notebook has changes the text has not, and hands the new text over at once", async () => {
    vi.useFakeTimers()
    const model = new FakeNotebook(notebook([code("a", "a = 1"), code("b", "b = 2")]))
    const persist = vi.fn(async () => true)
    const { store, heard } = followed(model, persist)
    const text = store.state()?.text ?? ""

    model.edit(1, "b = 3")
    await expect(store.save(text.replace("a = 1", "a = 10"))).rejects.toThrow("The notebook changed, write again")
    expect(heard).toEqual(["# %%\na = 1\n\n# %%\nb = 3\n"])
    expect(model.cells.map((cell) => cell.source)).toEqual(["a = 1", "b = 3"])
    expect(persist).not.toHaveBeenCalled()
    vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toHaveLength(1)

    await expect(store.save(heard[0].replace("a = 1", "a = 10"))).resolves.toBe("# %%\na = 10\n\n# %%\nb = 3\n")
  })

  it("never hands out the text being saved while the persist runs", async () => {
    vi.useFakeTimers()
    const model = new FakeNotebook(notebook([code("a", "a = 1")]))
    const persisted = Promise.withResolvers<boolean>()
    const { store, heard } = followed(model, () => persisted.promise)
    const saving = store.save("# %%\na = 2\n")
    vi.advanceTimersByTime(2 * NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toEqual([])
    persisted.resolve(true)
    await expect(saving).resolves.toBe("# %%\na = 2\n")
    expect(heard).toEqual([])
  })

  it("watches the model only while bound and followed", () => {
    const model = new FakeNotebook(notebook([code("a", "a = 1")]))
    const on = vi.spyOn(model.ydoc, "on")
    const off = vi.spyOn(model.ydoc, "off")
    const store = createNotebookTextStore()
    const unbind = store.bind(model, async () => true, () => {})
    expect(on).not.toHaveBeenCalled()

    const stop = store.follow(() => {})
    expect(on).toHaveBeenCalledTimes(1)
    stop()
    expect(off).toHaveBeenCalledTimes(1)

    store.follow(() => {})
    expect(on).toHaveBeenCalledTimes(2)
    unbind()
    expect(off).toHaveBeenCalledTimes(2)
    expect(store.state()).toBeNull()
  })

  it("catches a new follower up at once with changes made while nobody followed", () => {
    const model = new FakeNotebook(notebook([code("a", "a = 1")]))
    const store = createNotebookTextStore()
    store.bind(model, async () => true, () => {})
    const notified = vi.fn()
    store.subscribe(notified)

    model.edit(0, "a = 2")
    expect(store.state()?.text).toBe("# %%\na = 1\n")
    const heard: string[] = []
    store.follow((text) => heard.push(text))
    expect(heard).toEqual(["# %%\na = 2\n"])
    expect(store.state()?.text).toBe("# %%\na = 2\n")
    expect(notified).toHaveBeenCalledTimes(1)
  })

  it("keeps its followers across an unbind and a rebind and ignores a stale unbind", () => {
    vi.useFakeTimers()
    const first = new FakeNotebook(notebook([code("a", "a = 1")]))
    const second = new FakeNotebook(notebook([code("b", "b = 1")]))
    const store = createNotebookTextStore()
    const heard: string[] = []
    store.follow((text) => heard.push(text))
    const unbindFirst = store.bind(first, async () => true, () => {})
    first.edit(0, "a = 2")
    unbindFirst()
    vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toEqual([])

    store.bind(second, async () => true, () => {})
    const state = store.state()
    expect(state?.text).toBe("# %%\nb = 1\n")
    unbindFirst()
    expect(store.state()).toBe(state)
    first.edit(0, "a = 3")
    second.edit(0, "b = 2")
    vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    expect(heard).toEqual(["# %%\nb = 2\n"])
  })

  it("runs nothing before a save", () => {
    const run = vi.fn()
    const store = createNotebookTextStore()
    store.bind(new FakeNotebook(notebook([code("a", "a = 1")])), async () => true, run)
    store.runAt(1)
    expect(run).not.toHaveBeenCalled()
  })

  it("falls back to .py for an extension the backend refuses", () => {
    const extensionOf = (file_extension: JsonObject[string]) => {
      const store = createNotebookTextStore()
      store.bind(new FakeNotebook({ ...notebook([]), metadata: { language_info: { file_extension } } }), async () => true, () => {})
      return store.state()?.extension
    }
    expect(extensionOf(".jl")).toBe(".jl")
    for (const invalid of ["", "py", ".c++", ".toolongext", 3]) expect(extensionOf(invalid)).toBe(".py")
  })
})
