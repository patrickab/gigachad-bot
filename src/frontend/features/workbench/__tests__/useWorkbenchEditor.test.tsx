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
const follow = vi.hoisted(() => ({ files: [] as readonly string[], changed: null as ((path: string) => void) | null }))
vi.mock("@/hooks/useFollowFiles", () => ({
  useFollowFiles: (files: readonly string[], onChange: (path: string) => void) => {
    follow.files = files
    follow.changed = onChange
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

  it("pushes an outside change of an open file into Neovim, unless it matches the synced text", async () => {
    const { synced } = await openEditor()
    expect(follow.files).toEqual([MAIN])

    await act(async () => follow.changed?.(MAIN))
    expect(editor.ops.setContent).not.toHaveBeenCalled()

    api.loadFileViewerText.mockResolvedValueOnce("agent edit")
    await act(async () => follow.changed?.(MAIN))
    expect(editor.ops.setContent).toHaveBeenCalledExactlyOnceWith("agent edit", MAIN)
    expect(synced().get(MAIN)).toBe("agent edit")
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
})
