import { afterEach, describe, expect, it, vi } from "vitest"
import { createRef } from "react"
import { act, cleanup, render } from "@testing-library/react"
import { TabManager, type TabManagerHandle } from "@/features/navigation/TabManager"
import { nextTab } from "@/features/navigation/tabs"
import { useTab, type TabContextValue, type WorkspaceItem } from "@/contexts/TabContext"

vi.mock("@/lib/backend/syncStream", () => ({ subscribeToChanges: () => () => {} }))
vi.mock("@/contexts/ProjectContext", () => ({ useDisplayName: () => () => undefined }))

const doc: WorkspaceItem = { kind: "document", path: "/work/a.md", projectSlug: null }

afterEach(cleanup)

describe("a file closed in its tab", () => {
  function mount() {
    const manager = createRef<TabManagerHandle>()
    const probe: { tab?: TabContextValue } = {}
    function Probe() {
      const tab = useTab()
      if (tab.isActive) probe.tab = tab
      return null
    }
    render(<TabManager ref={manager}><Probe /></TabManager>)
    return { manager, probe }
  }

  it("reopens a closed file in its tab", () => {
    const { manager, probe } = mount()
    act(() => manager.current!.showItem(doc))
    act(() => probe.tab!.setAppMode("chat"))
    expect(probe.tab!.tab.appMode).toBe("chat")
    act(() => manager.current!.showItem(doc))
    expect(probe.tab!.tab.appMode).toBe("document")
    expect(manager.current!.getTabs()).toHaveLength(1)
  })

  it("reopens a closed file in its tab from another tab", () => {
    const { manager, probe } = mount()
    act(() => manager.current!.showItem(doc))
    const first = probe.tab!.tab.id
    act(() => probe.tab!.setAppMode("chat"))
    act(() => manager.current!.openTab(nextTab()))
    act(() => manager.current!.showItem(doc))
    expect(probe.tab!.tab.id).toBe(first)
    expect(probe.tab!.tab.appMode).toBe("document")
    expect(manager.current!.getTabs()).toHaveLength(2)
  })
})
