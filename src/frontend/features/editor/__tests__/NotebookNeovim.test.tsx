import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render } from "@testing-library/react"
import { useEffect } from "react"
import { announceFileMove } from "@/lib/backend/fileEvents"
import type { NeovimDocument, NeovimSelection } from "../NeovimEditor"
import { NotebookWithNeovim } from "../NotebookNeovim"

const seen = vi.hoisted(() => ({ documents: [] as (NeovimDocument | null)[], mounts: [] as string[], options: [] as { askKeymap?: boolean; onSelection?: (selection: NeovimSelection) => void }[] }))
const session = vi.hoisted(() => ({ terminal: null, status: { state: "starting" }, restart: () => {}, setCells: () => true, focusCell: () => true }))

vi.mock("@/lib/backend/syncStream", () => ({ subscribeToChanges: () => () => {} }))
vi.mock("@/contexts/SettingsContext", () => ({ useNeovimSetting: () => ({ enabled: true }) }))
vi.mock("../NeovimEditor", () => ({
  NeovimEditor: () => null,
  useNeovimTerminal: (document: NeovimDocument | null, _save: unknown, _quit: unknown, options: (typeof seen.options)[number]) => {
    seen.documents.push(document)
    seen.options.push(options)
    return session
  },
}))
// The notebook editor as a component that notes every mount with its path
vi.mock("next/dynamic", () => ({
  default: () => function NotebookEditor({ path }: { path: string }) {
    useEffect(() => { seen.mounts.push(path) }, []) // eslint-disable-line react-hooks/exhaustive-deps -- once per mount
    return null
  },
}))

afterEach(() => {
  cleanup()
  seen.documents = []
  seen.mounts = []
  seen.options = []
})

describe("NotebookWithNeovim", () => {
  it("gives another notebook opened in the tab its own session and editor, a renamed notebook keeps both", () => {
    const view = render(<NotebookWithNeovim path="/p/01.ipynb" tabId="tab" />)
    expect(seen.documents.at(-1)).toMatchObject({ key: "tab:0", path: "/p/01.ipynb" })

    view.rerender(<NotebookWithNeovim path="/p/02.ipynb" tabId="tab" />)
    expect(seen.documents.at(-1)).toMatchObject({ key: "tab:1", path: "/p/02.ipynb" })
    expect(seen.mounts).toEqual(["/p/01.ipynb", "/p/02.ipynb"])

    act(() => announceFileMove({ from: "/p/02.ipynb", to: "/p/renamed.ipynb" }))
    view.rerender(<NotebookWithNeovim path="/p/renamed.ipynb" tabId="tab" />)
    expect(seen.documents.at(-1)).toMatchObject({ key: "tab:1", path: "/p/renamed.ipynb" }) // same key: `renamed` names the buffer after the new file
    expect(seen.mounts).toEqual(["/p/01.ipynb", "/p/02.ipynb"])
  })

  it("asks Neovim for the ask keymap only for a notebook that takes selections, and hands them on", () => {
    const view = render(<NotebookWithNeovim path="/p/01.ipynb" tabId="tab" />)
    expect(seen.options.at(-1)?.askKeymap).toBe(false)

    const onSelection = vi.fn()
    view.rerender(<NotebookWithNeovim path="/p/01.ipynb" tabId="tab" onSelection={onSelection} />)
    expect(seen.options.at(-1)?.askKeymap).toBe(true)
    const selection = { path: "/p/01.ipynb.py", start_line: 2, end_line: 3, text: "x\ny" }
    seen.options.at(-1)?.onSelection?.(selection)
    expect(onSelection).toHaveBeenCalledWith(selection)
  })
})
