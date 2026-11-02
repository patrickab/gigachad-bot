import { afterEach, describe, expect, it, vi } from "vitest"
import { createRef, useContext } from "react"
import { act, cleanup, render } from "@testing-library/react"
import { TabManager, type TabManagerHandle } from "@/features/navigation/TabManager"
import { TabStripContext, filesOfTabs, nextTab, type TabStripValue } from "@/features/navigation/tabs"
import type { Tab, WorkspaceItem } from "@/contexts/TabContext"
import { onFileClosed } from "@/lib/backend/fileEvents"

vi.mock("@/lib/backend/syncStream", () => ({ subscribeToChanges: () => () => {} }))
vi.mock("@/contexts/ProjectContext", () => ({ useDisplayName: () => () => undefined }))

const NOTEBOOK = "/work/a.ipynb"
const tabShowing = (item: WorkspaceItem): Tab => ({ ...nextTab(), appMode: "document", workspaceItem: item })
const notebookTab = (path = NOTEBOOK) => tabShowing({ kind: "document", path, projectSlug: null, rootFile: true })

afterEach(cleanup)

describe("filesOfTabs", () => {
  it("lists the files of file items, whether or not the tab shows them", () => {
    const shown = notebookTab()
    const hidden = { ...notebookTab("/work/b.ipynb"), appMode: "chat" as const }
    const chat = nextTab()
    const learn = tabShowing({ kind: "learn", path: "", projectSlug: null })
    expect([...filesOfTabs([shown, hidden, chat, learn])]).toEqual([NOTEBOOK, "/work/b.ipynb"])
  })
})

describe("files closed by tabs", () => {
  /** A tab manager with the closed files collected, and the strip's close action. */
  function mount() {
    const manager = createRef<TabManagerHandle>()
    const closed: string[] = []
    const stop = onFileClosed((path) => closed.push(path))
    let strip!: TabStripValue
    function Probe() {
      strip = useContext(TabStripContext)!
      return null
    }
    render(<TabManager ref={manager}><Probe /></TabManager>)
    return { manager, closed, stop, close: (id: string) => act(() => strip.close(id)) }
  }

  it("announces a file when the last tab holding it closes", () => {
    const { manager, closed, stop, close } = mount()
    const first = notebookTab()
    const second = notebookTab()
    act(() => { manager.current!.openTab(first); manager.current!.openTab(second) })
    close(first.id)
    expect(closed).toEqual([])
    close(second.id)
    expect(closed).toEqual([NOTEBOOK])
    stop()
  })

  it("announces the file a tab leaves for another item, not the one it keeps", () => {
    const { manager, closed, stop } = mount()
    const tab = notebookTab()
    act(() => manager.current!.openTab(tab))
    act(() => manager.current!.switchToTab(tab.id))
    act(() => manager.current!.showItem({ kind: "document", path: "/work/b.ipynb", projectSlug: null, rootFile: true }))
    expect(closed).toEqual([NOTEBOOK])
    stop()
  })

  it("announces the old path of a renamed file, which the backend moved the kernel away from", () => {
    const { manager, closed, stop } = mount()
    act(() => manager.current!.openTab(notebookTab()))
    act(() => manager.current!.relocateDocument(NOTEBOOK, "/work/renamed.ipynb"))
    expect(closed).toEqual([NOTEBOOK])
    expect([...filesOfTabs(manager.current!.getTabs())]).toEqual(["/work/renamed.ipynb"])
    stop()
  })
})
