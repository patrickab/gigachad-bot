import { afterEach, describe, expect, it, vi } from "vitest"
import { createRef, useContext } from "react"
import { act, cleanup, render } from "@testing-library/react"
import { TabManager, type TabManagerHandle } from "@/features/navigation/TabManager"
import { TabStripContext, filesOfTabs, nextTab, type TabStripValue } from "@/features/navigation/tabs"
import { showingItem, useTab, type Tab, type TabContextValue, type WorkspaceItem } from "@/contexts/TabContext"
import { onFileClosed } from "@/lib/backend/fileEvents"

vi.mock("@/lib/backend/syncStream", () => ({ subscribeToChanges: () => () => {} }))
vi.mock("@/contexts/ProjectContext", () => ({ useDisplayName: () => () => undefined }))

const NOTEBOOK = "/work/a.ipynb"
const tabShowing = (item: WorkspaceItem): Tab => ({ ...nextTab(), ...showingItem(item) })
const notebookTab = (path = NOTEBOOK) => tabShowing({ kind: "document", path, projectSlug: null })

afterEach(cleanup)

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
    act(() => manager.current!.showItem({ kind: "document", path: "/work/b.ipynb", projectSlug: null }))
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

describe("chrome", () => {
  it("sees the tab in front and acts on it alone, also after that tab opened another chat", () => {
    const manager = createRef<TabManagerHandle>()
    let front!: TabContextValue
    function Probe() {
      front = useTab()
      return null
    }
    sessionStorage.clear()
    render(<TabManager ref={manager} chrome={(stack) => <><Probe />{stack}</>}>{null}</TabManager>)
    const [first] = manager.current!.getTabs()
    const second = nextTab()
    act(() => manager.current!.openTab(second))
    expect(front.tab.id).toBe(second.id)
    act(() => manager.current!.switchToTab(first.id))
    expect(front.tab.id).toBe(first.id)
    act(() => manager.current!.showChat("p/a.json"))
    act(() => front.updateConfig({ selectedModel: "probe" }))
    const configs = manager.current!.getTabs().map((tab) => tab.config.selectedModel)
    expect(configs).toEqual(["probe", second.config.selectedModel])
    expect(front.tab.historyFile).toBe("p/a.json")
  })
})
