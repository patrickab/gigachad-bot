import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render, screen } from "@testing-library/react"
import { useEffect, type ComponentProps, type ReactNode } from "react"

const neovim = vi.hoisted(() => ({ enabled: true }))
vi.mock("@/contexts/SettingsContext", () => ({ useNeovimSetting: () => ({ available: true, enabled: neovim.enabled, setPreferred: () => {} }) }))
vi.mock("@/features/editor/NeovimEditor", () => ({ NeovimEditor: () => <p>neovim</p> }))
vi.mock("../WorkbenchChat", () => ({ WorkbenchChat: () => null, useWorkbenchChat: () => ({ quote: () => {} }) }))
vi.mock("../useFollowBuffers", () => ({ useFollowBuffers: () => {} }))
type EditorOptions = { onForwardSearch?: (path: string, line: number) => void; onNotice: (notice: string | null) => void }
const editor = vi.hoisted(() => ({ reveal: vi.fn(), send: vi.fn(() => true), focus: vi.fn(), options: null as EditorOptions | null }))
vi.mock("../useWorkbenchEditor", () => ({
  useWorkbenchEditor: (options: EditorOptions) => {
    editor.options = options
    return {
      terminal: { focus: editor.focus }, status: { state: "running" }, restart: () => {}, content: "", loadError: null, shownText: "",
      synced: { current: new Map() }, openPaths: [], setContent: () => true, send: editor.send, reveal: editor.reveal,
    }
  },
}))
// The frame shows its view, its editor, its pane, its status and its `center` controls only
const frame = vi.hoisted(() => ({ changeView: null as ((view: WorkbenchView) => void) | null }))
vi.mock("../WorkbenchFrame", () => ({
  WorkbenchFrame: ({ view, onViewChange, editor, center, status, renderPane }: {
    view: WorkbenchView
    onViewChange: (view: WorkbenchView) => void
    editor: ReactNode
    center?: ReactNode
    status?: ReactNode
    renderPane: (layout: unknown) => ReactNode
  }) => {
    frame.changeView = onViewChange
    return <div data-testid="frame" data-view={view}>{editor}{renderPane({ width: 600, top: 56, bottom: 12, height: 500 })}{status}{center}</div>
  },
  EditorNote: ({ children }: { children: ReactNode }) => <p>{children}</p>,
  WorkbenchNotice: ({ children }: { children: ReactNode }) => <span role="alert">{children}</span>,
}))

import { Workbench, type WorkbenchPane } from "../Workbench"
import { useWorkbenchNotice, type ReportNotice } from "../useWorkbenchNotice"
import type { WorkbenchView } from "../WorkbenchSplit"

const CHAPTER = "/home/me/talk/chapter.tex"

type WorkbenchTestProps = Partial<ComponentProps<typeof Workbench>>

/** The workbench of a LaTeX file with its own notice channel, the props the test does not read left out. */
function TestWorkbench(props: WorkbenchTestProps) {
  const notices = useWorkbenchNotice()
  return <Workbench {...{
    path: "/home/me/talk/main.tex",
    cwd: "/home/me/talk",
    quoteLanguage: "latex",
    renderPane: () => null,
    notices,
    ...props,
  } as unknown as ComponentProps<typeof Workbench>} />
}

const workbenchOf = (props: WorkbenchTestProps = {}) => <TestWorkbench {...props} />

/** The workbench, its latest pane kept in `rendered.pane`. */
function renderWorkbench() {
  const rendered: { pane: WorkbenchPane | null } = { pane: null }
  render(workbenchOf({ renderPane: (pane) => { rendered.pane = pane; return null } }))
  return rendered
}

const view = () => screen.getByTestId("frame").dataset.view

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  neovim.enabled = true
})

describe("Workbench", () => {
  it.each([{ enabled: true }, { enabled: false }])("offers the center controls and the editor to the pane only while Neovim is enabled ($enabled)", ({ enabled }) => {
    neovim.enabled = enabled
    const rendered: { pane: WorkbenchPane | null } = { pane: null }
    render(workbenchOf({
      renderPane: (pane) => { rendered.pane = pane; return null },
      center: () => <button type="button">Insert</button>,
    }))

    expect(screen.queryByRole("button", { name: "Insert" }) !== null).toBe(enabled)
    expect(rendered.pane?.editor != null).toBe(enabled)
  })

  it("reveals a line and column through the editor and, to focus Neovim, brings its hidden page back before focusing it", () => {
    const rendered = renderWorkbench()
    act(() => frame.changeView?.("render"))
    const viewsAtFocus: (string | undefined)[] = []
    editor.focus.mockImplementation(() => viewsAtFocus.push(view()))

    act(() => rendered.pane?.editor?.reveal(CHAPTER, 12, 5, { focus: true }))

    expect(editor.reveal).toHaveBeenCalledExactlyOnceWith(CHAPTER, 12, 5)
    expect(viewsAtFocus).toEqual(["split"])
  })

  it("puts the cursor through the editor without showing or focusing Neovim", () => {
    const rendered = renderWorkbench()
    act(() => frame.changeView?.("render"))

    act(() => rendered.pane?.editor?.reveal(CHAPTER, 12))

    expect(editor.reveal).toHaveBeenCalledExactlyOnceWith(CHAPTER, 12, undefined)
    expect(view()).toBe("render")
    expect(editor.focus).not.toHaveBeenCalled()
  })

  it.each([{ moved: true, shown: "split" }, { moved: false, shown: "editor" }])("hands <leader>v to onForwardSearch and brings the pane back from the editor-only view when it moved there ($moved)", async ({ moved, shown }) => {
    const onForwardSearch = vi.fn(async () => moved)
    render(workbenchOf({ onForwardSearch }))
    act(() => frame.changeView?.("editor"))

    await act(async () => editor.options?.onForwardSearch?.(CHAPTER, 12))

    expect(onForwardSearch).toHaveBeenCalledExactlyOnceWith(CHAPTER, 12)
    expect(view()).toBe(shown)
  })

  it("shows the latest notice of the owner or the editor, a success of its source clears it and one of another source does not", () => {
    const owner: { report: ReportNotice | null } = { report: null }
    function Owner() {
      const notices = useWorkbenchNotice()
      useEffect(() => { owner.report = notices.report })
      return workbenchOf({ notices })
    }
    render(<Owner />)
    const notice = () => screen.queryByRole("alert")?.textContent ?? null

    act(() => editor.options?.onNotice("Not UTF-8"))
    expect(notice()).toBe("Not UTF-8")
    act(() => owner.report?.("lookup", "No match"))
    expect(notice()).toBe("No match")
    act(() => editor.options?.onNotice(null))
    expect(notice()).toBe("No match")
    act(() => owner.report?.("lookup", null))
    expect(notice()).toBeNull()
  })
})
