import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook } from "@testing-library/react"
import { NOTEBOOK_SAVE_DELAY_MS, useNotebookFile } from "../useNotebookFile"
import { rejoinNotebook, serializeNotebook, type NotebookCellContent, type NotebookContent } from "../notebookSync"
import { FakeNotebook } from "./fakeNotebook"

// A file store the api calls read and write
const world = vi.hoisted(() => {
  class ApiError extends Error {
    constructor(message: string, readonly status: number) {
      super(message)
    }
  }
  return {
    ApiError,
    files: new Map<string, string>(),
    reads: [] as string[],
    writes: [] as { path: string; text: string }[],
    /** Holds every write open until resolved, to see what happens while one is in flight. */
    hold: null as null | (() => Promise<void>),
    failWrites: null as null | Error,
    changed: null as ((path: string) => void) | null,
  }
})
vi.mock("@/lib/backend/api", () => ({
  ApiError: world.ApiError,
  loadFileViewerText: async (path: string) => {
    world.reads.push(path)
    const text = world.files.get(path)
    if (text === undefined) throw new world.ApiError("missing", 404)
    return text
  },
  writeRootFile: async (path: string, text: string) => {
    if (world.failWrites) throw world.failWrites
    // The file changes when the write starts, its reply comes later
    world.files.set(path, text)
    world.writes.push({ path, text })
    await world.hold?.()
  },
}))
vi.mock("@/lib/backend/syncStream", () => ({ syncCursorReady: () => Promise.resolve(), subscribeToChanges: () => () => {} }))
vi.mock("@/hooks/useFollowFiles", () => ({
  useFollowFiles: (_files: readonly string[], onChange: (path: string) => void) => {
    world.changed = onChange
  },
}))

const PATH = "/work/analysis.ipynb"
const code = (id: string, source: string, extra: Partial<NotebookCellContent> = {}): NotebookCellContent => ({ cell_type: "code", id, source, metadata: {}, outputs: [], execution_count: null, ...extra })
const notebook = (...cells: NotebookCellContent[]): NotebookContent => ({ cells, metadata: { kernelspec: { name: "python3" } }, nbformat: 4, nbformat_minor: 5 })
/** The file text for `content` as an earlier save of Jupyter would have written it. */
const fileText = (content: NotebookContent) => serializeNotebook(content)

const START = notebook(code("a", "a = 1"), code("b", "b = 1"))

/** Run all timers and promises due within `ms`. */
const elapse = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms) })
const settle = () => elapse(0)
/** Tell the hook the followed file changed, as the file watcher does. */
const announce = (path = PATH) => act(async () => { world.changed?.(path); await vi.advanceTimersByTimeAsync(0) })

async function open(path = PATH, content: NotebookContent = START) {
  world.files.set(path, fileText(content))
  const hook = renderHook(({ at }) => useNotebookFile(at), { initialProps: { at: path } })
  await settle()
  const model = new FakeNotebook(hook.result.current.initial as NotebookContent)
  act(() => hook.result.current.bindModel(model))
  return { hook, model }
}

beforeEach(() => {
  vi.useFakeTimers()
  world.files.clear()
  world.reads.length = 0
  world.writes.length = 0
  world.hold = null
  world.failWrites = null
  world.changed = null
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("load", () => {
  it("loads the notebook with its text as strings", async () => {
    world.files.set(PATH, fileText(START).replace('"a = 1"', '"a = 1\\n",\n    "a2"'))
    const { result } = renderHook(() => useNotebookFile(PATH))
    expect(result.current.initial).toBeNull()
    await settle()
    expect(result.current.error).toBeNull()
    expect(result.current.initial).toEqual(rejoinNotebook(JSON.parse(world.files.get(PATH) as string) as NotebookContent))
    expect(typeof result.current.initial?.cells[0].source).toBe("string")
  })

  it("loads a blank file as an empty notebook", async () => {
    world.files.set(PATH, "")
    const { result } = renderHook(() => useNotebookFile(PATH))
    await settle()
    expect(result.current.initial).toEqual({ cells: [], metadata: {}, nbformat: 4, nbformat_minor: 5 })
  })

  it("reports a file that cannot be read or is not a notebook, and loads again on reload", async () => {
    const { result } = renderHook(() => useNotebookFile(PATH))
    await settle()
    expect(result.current.initial).toBeNull()
    expect(result.current.error).toMatch(/no longer exists/)

    world.files.set(PATH, "{ not json")
    act(() => { result.current.reload() })
    await settle()
    expect(result.current.error).toMatch(/not a valid notebook/)

    world.files.set(PATH, fileText(START))
    act(() => { result.current.reload() })
    await settle()
    expect(result.current.error).toBeNull()
    expect(result.current.initial?.cells).toHaveLength(2)
  })
})

describe("save", () => {
  it("writes after the model has been quiet, in the file's style", async () => {
    const original = world.files.set(PATH, fileText(START)).get(PATH) as string
    const { model } = await open()
    model.edit(0, "a = 2")
    await elapse(NOTEBOOK_SAVE_DELAY_MS - 1)
    expect(world.writes).toHaveLength(0)
    await elapse(1)
    expect(world.writes).toHaveLength(1)
    expect(world.writes[0].path).toBe(PATH)
    expect(world.writes[0].text).toBe(original.replace('"a = 1"', '"a = 2"'))
  })

  it("restarts the wait on every change", async () => {
    const { model } = await open()
    model.edit(0, "a = 2")
    await elapse(NOTEBOOK_SAVE_DELAY_MS - 100)
    model.edit(0, "a = 3")
    await elapse(NOTEBOOK_SAVE_DELAY_MS - 100)
    expect(world.writes).toHaveLength(0)
    await elapse(100)
    expect(world.writes).toHaveLength(1)
    expect(world.writes[0].text).toContain('"a = 3"')
  })

  it("does not write when the model holds what the file holds", async () => {
    const { model } = await open()
    model.edit(0, "a = 1")
    model.setMetadata("kernelspec", { name: "python3" })
    await elapse(NOTEBOOK_SAVE_DELAY_MS * 2)
    expect(world.writes).toHaveLength(0)
  })

  it("does not write what the library added to the model", async () => {
    world.files.set(PATH, fileText(START))
    const hook = renderHook(() => useNotebookFile(PATH))
    await settle()
    const model = new FakeNotebook(hook.result.current.initial as NotebookContent, { injectEditable: true })
    act(() => hook.result.current.bindModel(model))
    model.edit(1, "b = 1")
    await elapse(NOTEBOOK_SAVE_DELAY_MS * 2)
    expect(world.writes).toHaveLength(0)
    model.edit(1, "b = 2")
    await elapse(NOTEBOOK_SAVE_DELAY_MS)
    expect(world.writes).toHaveLength(1)
    expect(world.writes[0].text).not.toContain("editable")
  })

  it("writes at once on save() and not again for the same change", async () => {
    const { hook, model } = await open()
    model.edit(0, "a = 2")
    await act(() => hook.result.current.save())
    expect(world.writes).toHaveLength(1)
    await elapse(NOTEBOOK_SAVE_DELAY_MS * 2)
    expect(world.writes).toHaveLength(1)
  })

  it("never writes an older save after a newer one", async () => {
    let release: () => void = () => {}
    world.hold = () => new Promise<void>((resolve) => { release = resolve })
    const { hook, model } = await open()
    model.edit(0, "first")
    const first = hook.result.current.save()
    await settle()
    model.edit(0, "second")
    const second = hook.result.current.save()
    await settle()
    expect(world.writes.map((write) => write.text.includes('"first"'))).toEqual([true])
    world.hold = null
    await act(async () => { release(); await first; await second })
    expect(world.writes).toHaveLength(2)
    expect(world.writes[1].text).toContain('"second"')
  })

  it("reports a failed save() as saveError until a save works", async () => {
    const { hook, model } = await open()
    model.edit(0, "a = 2")
    world.failWrites = new Error("disk full")
    await act(() => hook.result.current.save())
    expect(hook.result.current.saveError).toBe("disk full")
    world.failWrites = null
    await act(() => hook.result.current.save())
    expect(hook.result.current.saveError).toBeNull()
    expect(world.writes).toHaveLength(1)
  })

  it("reports a failed autosave and writes the change on the next one", async () => {
    const { hook, model } = await open()
    world.failWrites = new Error("offline")
    model.edit(0, "a = 2")
    await elapse(NOTEBOOK_SAVE_DELAY_MS)
    expect(hook.result.current.saveError).toBe("offline")
    world.failWrites = null
    model.edit(1, "b = 2")
    await elapse(NOTEBOOK_SAVE_DELAY_MS)
    expect(hook.result.current.saveError).toBeNull()
    expect(world.writes[0].text).toContain('"a = 2"')
    expect(world.writes[0].text).toContain('"b = 2"')
  })

  it("saves what is pending when the model goes away", async () => {
    const { hook, model } = await open()
    model.edit(0, "a = 2")
    act(() => hook.result.current.bindModel(null))
    await settle()
    expect(world.writes).toHaveLength(1)
    model.edit(0, "a = 3")
    await elapse(NOTEBOOK_SAVE_DELAY_MS * 2)
    expect(world.writes).toHaveLength(1)
  })

  it("saves what is pending when the tab closes", async () => {
    const { hook, model } = await open()
    model.edit(0, "a = 2")
    hook.unmount()
    await settle()
    expect(world.writes).toHaveLength(1)
  })
})

describe("follow", () => {
  it("ignores the echo of its own write and keeps what was typed since", async () => {
    const { hook, model } = await open()
    model.edit(0, "a = 2")
    await act(() => hook.result.current.save())
    model.edit(0, "a = typed after the save")
    await announce()
    expect(model.cells[0].source).toBe("a = typed after the save")
    expect(model.transactions).toEqual([])
  })

  it("ignores the echo that arrives while the write is still open", async () => {
    let release: () => void = () => {}
    world.hold = () => new Promise<void>((resolve) => { release = resolve })
    const { hook, model } = await open()
    model.edit(0, "a = 2")
    const saving = hook.result.current.save()
    await settle()
    model.edit(0, "a = typed meanwhile")
    await announce()
    expect(model.cells[0].source).toBe("a = typed meanwhile")
    world.hold = null
    await act(async () => { release(); await saving })
    await announce()
    expect(model.cells[0].source).toBe("a = typed meanwhile")
    expect(model.transactions).toEqual([])
  })

  it("merges what someone else wrote into the model", async () => {
    const { model } = await open()
    model.edit(1, "b = typed")
    world.files.set(PATH, fileText(notebook(code("a", "a = agent"), code("b", "b = 1"), code("c", "c = agent", { outputs: [{ output_type: "stream", name: "stdout", text: "hi\n" }], execution_count: 1 }))))
    await announce()
    expect(model.cells.map((cell) => cell.source)).toEqual(["a = agent", "b = typed", "c = agent"])
    expect(model.cells[2].outputs).toEqual([{ output_type: "stream", name: "stdout", text: "hi\n" }])
    expect(model.cells[2].execution_count).toBe(1)
    expect(model.transactions.length).toBeGreaterThan(0)
    expect(model.transactions.every((undoable) => undoable === false)).toBe(true)
  })

  it("does not write the merged notebook back, only what the editor added to it", async () => {
    const { model } = await open()
    world.files.set(PATH, fileText(notebook(code("a", "a = agent"), code("b", "b = 1"))))
    await announce()
    await elapse(NOTEBOOK_SAVE_DELAY_MS * 2)
    expect(world.writes).toHaveLength(0)

    model.edit(1, "b = typed")
    await elapse(NOTEBOOK_SAVE_DELAY_MS)
    expect(world.writes).toHaveLength(1)
    expect(world.writes[0].text).toBe(fileText(notebook(code("a", "a = agent"), code("b", "b = typed"))))
  })

  it("merges a second change against the first", async () => {
    const { model } = await open()
    world.files.set(PATH, fileText(notebook(code("a", "a = agent"), code("b", "b = 1"))))
    await announce()
    model.edit(1, "b = typed")
    world.files.set(PATH, fileText(notebook(code("a", "a = agent again"), code("b", "b = 1"))))
    await announce()
    expect(model.cells.map((cell) => cell.source)).toEqual(["a = agent again", "b = typed"])
  })

  it("ignores a file that does not parse yet and takes the finished one", async () => {
    const { model } = await open()
    world.files.set(PATH, '{"cells": [')
    await announce()
    expect(model.transactions).toEqual([])
    world.files.set(PATH, fileText(notebook(code("a", "a = agent"), code("b", "b = 1"))))
    await announce()
    expect(model.cells[0].source).toBe("a = agent")
  })

  it("ignores a change of another file", async () => {
    const { model } = await open()
    world.files.set(PATH, fileText(notebook(code("a", "a = agent"), code("b", "b = 1"))))
    await announce("/work/other.ipynb")
    expect(model.cells[0].source).toBe("a = 1")
  })

  it("catches a model up with what changed before it was bound", async () => {
    world.files.set(PATH, fileText(START))
    const hook = renderHook(() => useNotebookFile(PATH))
    await settle()
    const shown = hook.result.current.initial as NotebookContent
    world.files.set(PATH, fileText(notebook(code("a", "a = agent"), code("b", "b = 1"))))
    await announce()
    const model = new FakeNotebook(shown)
    act(() => hook.result.current.bindModel(model))
    expect(model.cells[0].source).toBe("a = agent")
    await elapse(NOTEBOOK_SAVE_DELAY_MS * 2)
    expect(world.writes).toHaveLength(0)
  })
})

describe("rename", () => {
  const RENAMED = "/work/renamed.ipynb"

  it("moves reads, writes and following to the new path without a reload", async () => {
    const { hook, model } = await open()
    const readsBefore = world.reads.length
    world.files.set(RENAMED, world.files.get(PATH) as string)
    world.files.delete(PATH)
    const initial = hook.result.current.initial
    hook.rerender({ at: RENAMED })
    await settle()

    expect(world.reads).toHaveLength(readsBefore)
    expect(hook.result.current.initial).toBe(initial)

    model.edit(0, "a = after rename")
    await elapse(NOTEBOOK_SAVE_DELAY_MS)
    expect(world.writes.map((write) => write.path)).toEqual([RENAMED])
    expect(world.files.has(PATH)).toBe(false)
  })

  it("follows the new path and ignores the old one", async () => {
    const { hook, model } = await open()
    world.files.set(RENAMED, fileText(notebook(code("a", "a = agent"), code("b", "b = 1"))))
    hook.rerender({ at: RENAMED })
    await announce(PATH)
    expect(model.cells[0].source).toBe("a = 1")
    await announce(RENAMED)
    expect(model.cells[0].source).toBe("a = agent")
  })
})
