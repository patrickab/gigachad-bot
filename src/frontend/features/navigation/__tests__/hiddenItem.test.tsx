import { afterEach, describe, expect, it, vi } from "vitest"
import { createRef } from "react"
import { act, cleanup, render } from "@testing-library/react"
import { TabManager, type TabManagerHandle } from "@/features/navigation/TabManager"
import { nextTab } from "@/features/navigation/tabs"
import { tabSurface, useTab, type TabContextValue, type WorkspaceItem } from "@/contexts/TabContext"
import { announceFileRemoval, onFileClosed } from "@/lib/backend/fileEvents"

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
    act(() => probe.tab!.hideItem())
    expect(tabSurface(probe.tab!.tab)).toBe("chat")
    act(() => manager.current!.showItem(doc))
    expect(tabSurface(probe.tab!.tab)).toBe("document")
    expect(manager.current!.getTabs()).toHaveLength(1)
  })

  it("reopens a closed file in its tab from another tab", () => {
    const { manager, probe } = mount()
    act(() => manager.current!.showItem(doc))
    const first = probe.tab!.tab.id
    act(() => probe.tab!.hideItem())
    act(() => manager.current!.openTab(nextTab()))
    act(() => manager.current!.showItem(doc))
    expect(probe.tab!.tab.id).toBe(first)
    expect(tabSurface(probe.tab!.tab)).toBe("document")
    expect(manager.current!.getTabs()).toHaveLength(2)
  })
  it("deleting a hidden file keeps the tab", () => {
    const { manager, probe } = mount()
    const closed = vi.fn()
    const off = onFileClosed(closed)
    act(() => manager.current!.showItem(doc))
    const id = probe.tab!.tab.id
    act(() => probe.tab!.hideItem())
    act(() => announceFileRemoval(doc.path))
    off()
    expect(manager.current!.getTabs()).toHaveLength(1)
    expect(probe.tab!.tab).toMatchObject({ id, workspaceItem: null })
    expect(closed).toHaveBeenCalledWith(doc.path)
  })

  it("deleting a shown file closes the tab", () => {
    const { manager, probe } = mount()
    act(() => manager.current!.showItem(doc))
    const id = probe.tab!.tab.id
    act(() => announceFileRemoval(doc.path))
    expect(manager.current!.getTabs()).toHaveLength(1)
    expect(probe.tab!.tab.id).not.toBe(id)
  })
})
