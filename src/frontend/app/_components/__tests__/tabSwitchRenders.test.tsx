import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render } from "@testing-library/react"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"
import { useTab, type Tab } from "@/contexts/TabContext"

/** A tab switch and a chat save re-render only the tabs they concern: counted on each tab's surface. */

const { renders, saved, branchMeta } = vi.hoisted(() => ({
  renders: new Map<string, number>(),
  saved: vi.fn(async () => {}),
  branchMeta: { current: {} as Record<string, unknown> },
}))

vi.mock("../WorkspaceSurface", () => ({
  WorkspaceSurface: function CountedSurface() {
    const { tab } = useTab()
    renders.set(tab.id, (renders.get(tab.id) ?? 0) + 1)
    return null
  },
}))
vi.mock("../TabSidebar", () => ({ TabSidebar: () => null }))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  listChatHistories: async () => ({ files: ["a.json", "b.json", "c.json"] }),
  fetchBranchMeta: async () => branchMeta.current,
  loadChatHistory: async () => ({ messages: [] }),
  saveChat: saved,
}))
vi.mock("@/hooks/useResource", () => ({
  useResource: () => undefined,
  useResourceState: () => ({ value: undefined, error: null }),
  useResourceFamily: () => ({}),
}))
const settings = { tabDefaults: DEFAULT_TAB_CONFIG, ocrModel: "" }
const project = { refreshProjects: async () => {} }
const undoDelete = { schedule: () => {}, pending: () => false }
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => settings,
}))
vi.mock("@/contexts/ProjectContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/ProjectContext")>(),
  useProject: () => project,
}))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => undoDelete }))
vi.mock("@/features/pi/sessionStore", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/features/pi/sessionStore")>(),
  PiSessionsProvider: ({ children }: { children: React.ReactNode }) => children,
}))
vi.mock("@/features/pi/notifications", () => ({ startPiNotifications: () => () => {} }))
vi.mock("@/features/navigation/ProviderUsage", () => ({ ProviderUsagePoll: () => null }))
vi.mock("@/features/memory/MemoryViewer", () => ({ MemoryViewer: () => null }))
vi.mock("@/features/editor/ShellTerminal", () => ({ ShellTerminals: () => null }))
vi.mock("@/features/ambient/AmbientBackdrop", () => ({ AmbientBackdrop: () => null }))
vi.mock("@/features/notebook/useKernelShutdown", () => ({ useKernelShutdown: () => {} }))
vi.mock("@/lib/backend/resources", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/resources")>(),
  watchDocumentLists: () => () => {},
}))

const { AppShell } = await import("../AppShell")

const tab = (id: string): Tab => ({ id, chatId: `chat-${id}`, historyFile: `${id}.json`, title: id, config: DEFAULT_TAB_CONFIG, appMode: "chat", projectSlug: null, workspaceItem: null })
const flush = () => act(async () => { for (let i = 0; i < 5; i++) await new Promise((resolve) => setTimeout(resolve, 0)) })

/** Renders per tab while `action` runs and settles. */
async function count(action: () => void): Promise<Record<string, number>> {
  renders.clear()
  await act(async () => action())
  await flush()
  return { a: renders.get("a") ?? 0, b: renders.get("b") ?? 0, c: renders.get("c") ?? 0 }
}

describe("tab switch and chat save renders", () => {
  beforeEach(async () => {
    sessionStorage.setItem("gigachad:tabs", JSON.stringify({ tabs: [tab("a"), tab("b"), tab("c")], activeId: "a" }))
    render(<AppShell />)
    await flush()
  })
  afterEach(() => {
    cleanup()
    branchMeta.current = {}
  })

  // Before the split branch contexts and the memoized TabContent: a switch rendered { a: 2, b: 2, c: 1 }, a save { a: 1, b: 1, c: 1 }
  it("a switch renders the two tabs it swaps, nothing else", async () => {
    expect(await count(() => fireEvent.keyDown(document.body, { key: "Tab" }))).toEqual({ a: 1, b: 1, c: 0 })
  })

  it("a save renders no tab", async () => {
    branchMeta.current = { "a.json": { chat_id: "chat-a", parent_id: null } }
    expect(await count(() => fireEvent.keyDown(window, { key: "s", altKey: true }))).toEqual({ a: 0, b: 0, c: 0 })
    expect(saved).toHaveBeenCalled()
  })
})
