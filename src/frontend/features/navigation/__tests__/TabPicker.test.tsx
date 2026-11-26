import { act, fireEvent, render, screen } from "@testing-library/react"
import { useSyncExternalStore } from "react"
import { describe, expect, it, vi } from "vitest"
import { TabContext, showingItem, type Tab, type TabContextValue } from "@/contexts/TabContext"
import { TabPicker } from "../TabPicker"
import { TabStripContext, nextTab, type TabStripValue } from "../tabs"

const store = vi.hoisted(() => ({ unread: new Set<string>(), listeners: new Set<() => void>() }))

vi.mock("@/features/pi/sessionStore", () => {
  const subscribe = (listener: () => void) => {
    store.listeners.add(listener)
    return () => store.listeners.delete(listener)
  }
  return {
    usePiSessionStatus: (sessionId: string | null) => {
      const unread = useSyncExternalStore(subscribe, () => sessionId !== null && store.unread.has(sessionId))
      return { running: false, waiting: false, unread, failed: false }
    },
  }
})

vi.mock("@/contexts/ProjectContext", () => ({
  useProject: () => ({ projects: [] }),
  useDisplayName: () => () => undefined,
}))

const piTab = (sessionId: string, title: string): Tab => ({
  ...nextTab(),
  title,
  ...showingItem({ kind: "pi", path: "repo", projectSlug: null, pi: { sessionId, repo: "repo" } }),
})

function setUnread(sessionId: string, unread: boolean) {
  act(() => {
    if (unread) store.unread.add(sessionId)
    else store.unread.delete(sessionId)
    for (const listener of store.listeners) listener()
  })
}

describe("TabPicker", () => {
  it("marks the title of a Pi tab whose session is unread", () => {
    const tabs = [piTab("s1", "Fix the parser"), piTab("s2", "Write docs")]
    const strip: TabStripValue = {
      tabs, activeTabId: tabs[0].id, labelOf: (tab) => tab.title ?? "", isLoading: () => false,
      select: vi.fn(), close: vi.fn(), add: vi.fn(() => ""), peek: vi.fn(),
    }
    const tab = { tab: tabs[0], isActive: true } as TabContextValue
    render(<TabStripContext.Provider value={strip}>
      <TabContext.Provider value={tab}><TabPicker /></TabContext.Provider>
    </TabStripContext.Provider>)
    fireEvent.keyDown(document.body, { key: " ", code: "Space" })
    fireEvent.keyUp(document.body, { key: " ", code: "Space" })

    const rowOf = (title: string) => screen.getByText(title).parentElement!
    expect(screen.queryByRole("img", { name: "Unread" })).toBeNull()
    setUnread("s2", true)
    expect(rowOf("Write docs").querySelector("[aria-label='Unread']")).not.toBeNull()
    expect(rowOf("Fix the parser").querySelector("[aria-label='Unread']")).toBeNull()
    setUnread("s2", false)
    expect(screen.queryByRole("img", { name: "Unread" })).toBeNull()
  })
})
