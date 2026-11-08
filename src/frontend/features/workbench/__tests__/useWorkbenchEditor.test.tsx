import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import type { NeovimDocument } from "@/features/editor/NeovimEditor"
import type { NeovimCommand, NeovimEventHandlers } from "@/features/editor/neovimProtocol"
import { announceFileMove } from "@/lib/backend/fileEvents"
import type { ChangeEvent } from "@/lib/backend/syncStream"

const api = vi.hoisted(() => ({
  ApiError: class ApiError extends Error { status = 500 },
  loadFileViewerText: vi.fn<(path: string) => Promise<string>>(),
  writeRootFile: vi.fn<(path: string, text: string) => Promise<void>>(async () => {}),
}))
vi.mock("@/lib/backend/api", () => api)
const feed = vi.hoisted(() => ({ listener: null as ((event: ChangeEvent) => void) | null }))
vi.mock("@/lib/backend/syncStream", () => ({
  syncCursorReady: () => Promise.resolve(),
  subscribeToChanges: (listener: (event: ChangeEvent) => void) => {
    feed.listener = listener
    return () => { feed.listener = null }
  },
}))

/** What the hook handed `useNeovimTerminal` on its latest render. */
interface Session {
  document: NeovimDocument | null
  save: (text: string, path: string) => Promise<string>
  options: { on: Required<Pick<NeovimEventHandlers, "selection" | "switched" | "open_failed">> }
}
const editor = vi.hoisted(() => ({
  session: null as Session | null,
  ops: {
    restart: vi.fn(),
    setContent: vi.fn<(text: string, path?: string) => boolean>(() => true),
    openFile: vi.fn<(path: string, line?: number, column?: number) => boolean>(() => true),
    renameFile: vi.fn<(from: string, to: string) => boolean>(() => true),
    insertLines: vi.fn<(text: string) => boolean>(() => true),
  },
}))
vi.mock("@/features/editor/NeovimEditor", () => ({
  useNeovimTerminal: (document: Session["document"], save: Session["save"], _onQuit: () => void, options: Session["options"]) => {
    editor.session = { document, save, options }
    // Each command goes to its own fake, so a test names the command it expects
    const { ops } = editor
    const send = (command: NeovimCommand) => {
      switch (command.type) {
        case "set_content": return ops.setContent(command.content, command.path)
        case "open_file": return command.line === undefined ? ops.openFile(command.path) : ops.openFile(command.path, command.line, command.column)
        case "renamed": return ops.renameFile(command.from, command.path)
        case "insert_lines": return ops.insertLines(command.text)
        default: return false
      }
    }
    return { terminal: null, status: { state: "running" }, restart: ops.restart, send }
  },
}))

import { useWorkbenchEditor } from "../useWorkbenchEditor"

const MAIN = "/home/me/thesis/main.tex"
const CHAPTER = "/home/me/thesis/chapters/intro.tex"
const RENAMED = "/home/me/thesis/chapters/opening.tex"
const textOf = (path: string) => `text of ${path}`
const moved = (from: string, to: string): ChangeEvent => ({ seq: 1, resource_kind: "document", resource_key: from, moved_to: to, version: null, operation: "move", device_id: "other" })

const callbacks = () => ({ onPathChange: vi.fn(), onSaved: vi.fn(), onClose: vi.fn(), onSelection: vi.fn(), onNotice: vi.fn() })

/** The editor over `MAIN`, loaded. `rerender({ path })` is the tab's item changing its path. */
async function openEditor(enabled = true) {
  api.loadFileViewerText.mockImplementation(async (path) => textOf(path))
  const props = callbacks()
  const view = renderHook(({ path }) => useWorkbenchEditor({ path, enabled, ...props }), { initialProps: { path: MAIN } })
  await waitFor(() => expect(view.result.current.content).not.toBeNull())
  const session = () => {
    if (!editor.session) throw new Error("No Neovim session")
    return editor.session
  }
  return { ...view, props, session, synced: () => view.result.current.synced.current }
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.clearAllMocks()
  editor.session = null
})

describe("useWorkbenchEditor", () => {
  it.each([{ enabled: true }, { enabled: false }])("loads the first file and starts the session with it once loaded only while Neovim is enabled ($enabled)", async ({ enabled }) => {
    api.loadFileViewerText.mockResolvedValue("\\documentclass{article}")
    const { result } = renderHook(() => useWorkbenchEditor({ path: MAIN, enabled, ...callbacks() }))
    expect(editor.session?.document).toBeNull()

    await waitFor(() => expect(result.current.content).toBe("\\documentclass{article}"))
    expect(editor.session?.document).toEqual(enabled ? expect.objectContaining({ key: MAIN, path: MAIN, content: "\\documentclass{article}" }) : null)
    expect(result.current.shownText).toBe("\\documentclass{article}")
  })

  it("says why the first file could not load", async () => {
    api.loadFileViewerText.mockRejectedValue(Object.assign(new api.ApiError("Not found"), { status: 404 }))
    const { result } = renderHook(() => useWorkbenchEditor({ path: MAIN, enabled: true, ...callbacks() }))
    await waitFor(() => expect(result.current.loadError).toBe("missing"))
  })

  it("switches Neovim's buffer when the path changes and loads each file once", async () => {
    const { result, rerender, session, synced } = await openEditor()
    expect(result.current.openPaths).toEqual([MAIN])
    rerender({ path: CHAPTER })
    await waitFor(() => expect(editor.ops.openFile).toHaveBeenCalledExactlyOnceWith(CHAPTER))
    expect(synced().get(CHAPTER)).toBe(textOf(CHAPTER))
    expect(result.current.openPaths).toEqual([MAIN, CHAPTER])
    expect(result.current.shownText).toBe(textOf(CHAPTER))
    // The session goes on with the first file as its document, a buffer switch never changes it
    expect(session().document).toMatchObject({ key: MAIN, path: MAIN })

    rerender({ path: MAIN })
    await waitFor(() => expect(editor.ops.openFile).toHaveBeenLastCalledWith(MAIN))
    expect(result.current.shownText).toBe(textOf(MAIN))
    expect(api.loadFileViewerText.mock.calls.map(([path]) => path)).toEqual([MAIN, CHAPTER])
  })

  it("tries again while Neovim's session is not open yet", async () => {
    const { rerender, synced } = await openEditor()
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })
    editor.ops.openFile.mockReturnValueOnce(false)

    rerender({ path: CHAPTER })
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(editor.ops.openFile).toHaveBeenCalledTimes(1)
    expect(synced().has(CHAPTER)).toBe(false)

    await act(async () => { await vi.advanceTimersByTimeAsync(250) })
    expect(editor.ops.openFile).toHaveBeenCalledTimes(2)
    expect(synced().get(CHAPTER)).toBe(textOf(CHAPTER))
  })

  it("puts the cursor on a line and column of the shown file at once", async () => {
    const { result, props } = await openEditor()

    act(() => result.current.reveal(MAIN, 42, 17))

    expect(editor.ops.openFile).toHaveBeenCalledExactlyOnceWith(MAIN, 42, 17)
    expect(props.onPathChange).not.toHaveBeenCalled()
    expect(props.onNotice).toHaveBeenLastCalledWith(null)
  })

  it("says so when the session cannot take a line of the shown file", async () => {
    const { result, props } = await openEditor()
    editor.ops.openFile.mockReturnValueOnce(false)

    act(() => result.current.reveal(MAIN, 42))

    expect(props.onNotice).toHaveBeenLastCalledWith("Neovim is not ready, the line was not shown")
  })

  it("switches to another file through the owner's path and puts the cursor on its line and column, also after a retry", async () => {
    const { result, rerender, props } = await openEditor()
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })
    editor.ops.openFile.mockReturnValueOnce(false)

    act(() => result.current.reveal(CHAPTER, 7, 3))
    expect(props.onPathChange).toHaveBeenCalledExactlyOnceWith(CHAPTER)
    expect(editor.ops.openFile).not.toHaveBeenCalled()

    rerender({ path: CHAPTER })
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    await act(async () => { await vi.advanceTimersByTimeAsync(250) })
    expect(editor.ops.openFile.mock.calls).toEqual([[CHAPTER, 7, 3], [CHAPTER, 7, 3]])

    // The line went with the switch, coming back later shows the file where Neovim left it
    rerender({ path: MAIN })
    rerender({ path: CHAPTER })
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(editor.ops.openFile).toHaveBeenLastCalledWith(CHAPTER)
  })

  it("reports the buffer Neovim switched to and does not switch back when the path follows", async () => {
    const { rerender, session, props } = await openEditor()
    act(() => session().options.on.switched({ type: "switched", path: CHAPTER }))
    expect(props.onPathChange).toHaveBeenCalledExactlyOnceWith(CHAPTER)

    rerender({ path: CHAPTER })
    act(() => session().options.on.switched({ type: "switched", path: CHAPTER }))
    expect(props.onPathChange).toHaveBeenCalledTimes(1)
    expect(editor.ops.openFile).not.toHaveBeenCalled()
  })

  it("keeps the shown file and says so when another file cannot be opened", async () => {
    const { rerender, props } = await openEditor()
    api.loadFileViewerText.mockRejectedValueOnce(new Error("Gone"))
    rerender({ path: CHAPTER })
    await waitFor(() => expect(props.onNotice).toHaveBeenLastCalledWith("Gone"))
    expect(props.onPathChange).toHaveBeenCalledExactlyOnceWith(MAIN)
    expect(editor.ops.openFile).not.toHaveBeenCalled()
  })

  it("goes back to the file Neovim still shows when the backend could not open the new one", async () => {
    const { rerender, session, synced, props } = await openEditor()
    rerender({ path: CHAPTER })
    await waitFor(() => expect(editor.ops.openFile).toHaveBeenCalledTimes(1))

    act(() => session().options.on.open_failed({ type: "open_failed", path: CHAPTER, message: "Not UTF-8" }))
    expect(props.onNotice).toHaveBeenLastCalledWith("Not UTF-8")
    expect(props.onPathChange).toHaveBeenCalledExactlyOnceWith(MAIN)
    expect(synced().has(CHAPTER)).toBe(false)

    // The owner's path follows, nothing tries the file again
    rerender({ path: MAIN })
    expect(editor.ops.openFile).toHaveBeenCalledTimes(1)
  })

  it("renames the buffer of a file this device or another one renamed, once, and follows it with the shown path", async () => {
    const { rerender, synced } = await openEditor()
    rerender({ path: CHAPTER })
    await waitFor(() => expect(editor.ops.openFile).toHaveBeenCalledTimes(1))

    act(() => announceFileMove({ from: CHAPTER, to: RENAMED }))
    expect(editor.ops.renameFile).toHaveBeenCalledExactlyOnceWith(CHAPTER, RENAMED)
    expect(synced().has(CHAPTER)).toBe(false)
    expect(synced().get(RENAMED)).toBe(textOf(CHAPTER))

    // The change feed echoes the rename and the tab's item takes the new path: neither renames or opens anything
    act(() => feed.listener?.(moved(CHAPTER, RENAMED)))
    rerender({ path: RENAMED })
    expect(editor.ops.renameFile).toHaveBeenCalledTimes(1)
    expect(editor.ops.openFile).toHaveBeenCalledTimes(1)

    // Another device's rename renames an open file too, and a file that is not open nothing
    act(() => feed.listener?.(moved("/home/me/thesis/other.tex", "/home/me/thesis/elsewhere.tex")))
    expect(editor.ops.renameFile).toHaveBeenCalledTimes(1)
    act(() => feed.listener?.(moved(MAIN, "/home/me/thesis/thesis.tex")))
    expect(editor.ops.renameFile).toHaveBeenLastCalledWith(MAIN, "/home/me/thesis/thesis.tex")
    expect(synced().get("/home/me/thesis/thesis.tex")).toBe(textOf(MAIN))
  })

  it("saves each file under the path Neovim names, one write at a time", async () => {
    const { session, synced, props } = await openEditor()
    const slow = Promise.withResolvers<void>()
    api.writeRootFile.mockReturnValueOnce(slow.promise)

    let first: Promise<string> = Promise.resolve("")
    let second: Promise<string> = Promise.resolve("")
    act(() => {
      first = session().save("one", MAIN)
      second = session().save("two", MAIN)
    })
    await waitFor(() => expect(api.writeRootFile).toHaveBeenCalledTimes(1))
    await act(async () => {
      slow.resolve()
      await Promise.all([first, second])
    })
    expect(api.writeRootFile.mock.calls).toEqual([[MAIN, "one"], [MAIN, "two"]])
    expect(synced().get(MAIN)).toBe("two")
    expect(props.onSaved).toHaveBeenLastCalledWith("two", MAIN)

    await act(() => session().save("hidden edit", CHAPTER))
    expect(api.writeRootFile).toHaveBeenLastCalledWith(CHAPTER, "hidden edit")
    expect(props.onSaved).toHaveBeenLastCalledWith("hidden edit", CHAPTER)
    expect(synced().has(CHAPTER)).toBe(false)
  })

  it("takes text for the rendered pane without Neovim when it is disabled", async () => {
    const { result, synced } = await openEditor(false)
    let taken = false
    act(() => { taken = result.current.setContent("agent edit", MAIN) })
    expect(taken).toBe(true)
    expect(editor.ops.setContent).not.toHaveBeenCalled()
    expect(synced().get(MAIN)).toBe("agent edit")
    expect(result.current.shownText).toBe("agent edit")
  })

  it("records the text Neovim took and leaves the record alone when it could not take it", async () => {
    const { result, synced } = await openEditor()
    editor.ops.setContent.mockReturnValueOnce(false)
    let taken = true
    act(() => { taken = result.current.setContent("agent edit", MAIN) })
    expect(taken).toBe(false)
    expect(synced().get(MAIN)).toBe(textOf(MAIN))

    act(() => { taken = result.current.setContent("agent edit", MAIN) })
    expect(taken).toBe(true)
    expect(editor.ops.setContent).toHaveBeenLastCalledWith("agent edit", MAIN)
    expect(synced().get(MAIN)).toBe("agent edit")
    expect(result.current.shownText).toBe("agent edit")
  })

  it("keeps the first file and shows the current one again after a restart", async () => {
    const { result, rerender, synced } = await openEditor()
    rerender({ path: CHAPTER })
    await waitFor(() => expect(editor.ops.openFile).toHaveBeenCalledTimes(1))
    act(() => { result.current.setContent("agent edit", MAIN) })

    act(() => result.current.restart())
    expect(editor.ops.restart).toHaveBeenCalledTimes(1)
    expect(synced().has(CHAPTER)).toBe(false)
    await waitFor(() => expect(editor.ops.openFile).toHaveBeenCalledTimes(2))
    expect(editor.ops.openFile).toHaveBeenLastCalledWith(CHAPTER)
    expect(editor.session?.document?.content).toBe("agent edit")
  })

  it("hands a command to Neovim, says whether it took it and reports the refusal until the next command goes in", async () => {
    const { result, props } = await openEditor()
    editor.ops.insertLines.mockReturnValueOnce(false)
    act(() => expect(result.current.send({ type: "insert_lines", text: "% note" }, "Neovim is not ready, the note was not added")).toBe(false))
    expect(props.onNotice).toHaveBeenLastCalledWith("Neovim is not ready, the note was not added")

    act(() => expect(result.current.send({ type: "insert_lines", text: "% note" }, "Neovim is not ready, the note was not added")).toBe(true))
    expect(editor.ops.insertLines).toHaveBeenLastCalledWith("% note")
    expect(props.onNotice).toHaveBeenLastCalledWith(null)
  })
})
