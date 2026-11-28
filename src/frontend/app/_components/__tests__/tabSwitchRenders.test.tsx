import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useEffect } from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"
import { useTab, type Tab } from "@/contexts/TabContext"
import { useBranches } from "@/features/navigation/BranchContext"
import type { Resource } from "@/lib/resource"

/** How often the chrome mounts, and which tabs a tab switch and a chat save re-render: counted on stubs of each piece. */

const { renders, mounts, saved, branchMeta, promptLists } = vi.hoisted(() => ({
  /** Renders by piece; the surface's by tab id. */
  renders: new Map<string, number>(),
  /** Instances mounted now, by piece. */
  mounts: new Map<string, number>(),
  saved: vi.fn(async () => {}),
  branchMeta: { current: {} as Record<string, unknown> },
  promptLists: vi.fn(async () => [{ slug: "coder", name: "Coder", includes: [] }]),
}))

function bump(counts: Map<string, number>, key: string, by = 1) {
  counts.set(key, (counts.get(key) ?? 0) + by)
}

function useMounted(name: string) {
  useEffect(() => {
    bump(mounts, name)
    return () => bump(mounts, name, -1)
  }, [name])
}

function counted<P extends object>(name: string, Real: (props: P) => React.ReactNode = () => null) {
  return function Counted(props: P) {
    useMounted(name)
    bump(renders, name)
    return <Real {...props} />
  }
}

vi.mock("../WorkspaceSurface", () => ({
  WorkspaceSurface: function CountedSurface() {
    const { tab } = useTab()
    useMounted("WorkspaceSurface")
    bump(renders, tab.id)
    return null
  },
}))
// The real sidebar, its renders counted by commits of its subtree: a render it causes itself counts too
vi.mock("@/features/navigation/Sidebar", async (importOriginal) => {
  const { Sidebar } = await importOriginal<typeof import("@/features/navigation/Sidebar")>()
  const { Profiler } = await import("react")
  return {
    Sidebar: function CountedSidebar(props: React.ComponentProps<typeof Sidebar>) {
      const branches = useBranches()
      useMounted("Sidebar")
      useEffect(() => bump(renders, "branches"), [branches])
      return <Profiler id="Sidebar" onRender={() => bump(renders, "Sidebar")}><Sidebar {...props} /></Profiler>
    },
  }
})
vi.mock("@/features/settings/SettingsMenu", () => ({
  SettingsMenu: ({ onEditPrompts }: { onEditPrompts: () => void }) => <button onClick={onEditPrompts}>Edit prompts</button>,
}))
vi.mock("@/features/navigation/TabPicker", () => ({ TabPicker: counted("TabPicker") }))
vi.mock("@/features/chat/ChatPicker", () => ({ ChatPicker: counted("ChatPicker") }))
vi.mock("@/features/pi/PiSessionPicker", () => ({ PiSessionPicker: counted("PiSessionPicker") }))
vi.mock("@/features/settings/PromptEditor", async (importOriginal) => ({
  PromptEditor: counted("PromptEditor", (await importOriginal<typeof import("@/features/settings/PromptEditor")>()).PromptEditor),
}))
vi.mock("@/components/ui/SaveAsModal", async (importOriginal) => ({
  SaveAsModal: counted("SaveAsModal", (await importOriginal<typeof import("@/components/ui/SaveAsModal")>()).SaveAsModal),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  listChatHistories: async () => ({ files: ["a.json", "b.json", "c.json"] }),
  fetchBranchMeta: async () => branchMeta.current,
  loadChatHistory: async () => ({ messages: [] }),
  saveChat: saved,
  fetchPromptList: promptLists,
  fetchPromptBlocks: async () => ({}),
  fetchPromptRaw: async () => ({ content: "old" }),
}))
// Only the chat histories load, which a save reads again.
vi.mock("@/hooks/useResource", async (importOriginal) => {
  const { useResourceState } = await importOriginal<typeof import("@/hooks/useResource")>()
  const { historiesResource } = await import("@/lib/backend/resources")
  return {
    useResource: () => undefined,
    useResourceState: (label: string | null, resource: Resource<unknown>) => useResourceState(label, resource, resource === historiesResource),
    useResourceFamily: () => ({}),
  }
})
const settings = { tabDefaults: DEFAULT_TAB_CONFIG, ocrModel: "" }
const project = { projects: [], refreshProjects: async () => {} }
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
vi.mock("@/features/navigation/ProviderUsage", () => ({ ProviderUsagePoll: () => null, ProviderUsage: () => null }))
vi.mock("@/features/memory/MemoryViewer", () => ({ MemoryViewer: () => null }))
vi.mock("@/features/editor/ShellTerminal", () => ({ ShellTerminals: () => null }))
vi.mock("@/features/ambient/AmbientBackdrop", () => ({ AmbientBackdrop: () => null }))
vi.mock("@/features/notebook/useKernelShutdown", () => ({ useKernelShutdown: () => {} }))
vi.mock("@/lib/backend/resources", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/resources")>(),
  watchDocumentLists: () => () => {},
}))

vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

const { AppShell } = await import("../AppShell")

const tab = (id: string): Tab => ({ id, chatId: `chat-${id}`, historyFile: `${id}.json`, title: id, config: DEFAULT_TAB_CONFIG, projectSlug: null, workspaceItem: null, itemHidden: false })
const flush = () => act(async () => { for (let i = 0; i < 5; i++) await new Promise((resolve) => setTimeout(resolve, 0)) })

/** Renders per tab's surface and of the sidebar, and changes of the branch context, while `action` runs and settles. */
async function count(action: () => void): Promise<Record<string, number>> {
  renders.clear()
  await act(async () => action())
  await flush()
  return { a: renders.get("a") ?? 0, b: renders.get("b") ?? 0, c: renders.get("c") ?? 0, sidebar: renders.get("Sidebar") ?? 0, branches: renders.get("branches") ?? 0 }
}

describe("chrome mounts, tab switch and chat save renders", () => {
  beforeEach(async () => {
    mounts.clear()
    sessionStorage.setItem("gigachad:tabs", JSON.stringify({ tabs: [tab("a"), tab("b"), tab("c")], activeId: "a" }))
    render(<AppShell />)
    await flush()
  })
  afterEach(() => {
    cleanup()
    branchMeta.current = {}
  })

  // Before the app's chrome every piece mounted 3 times, the sidebar until it joined the chrome.
  // The Save As dialog stays one per tab: it keeps that tab's typed name, and renders nothing while closed
  it("mounts the sidebar, pickers and prompt editor once, the surface and Save As dialog once per tab", () => {
    expect(Object.fromEntries(mounts)).toEqual({
      Sidebar: 1, TabPicker: 1, ChatPicker: 1, PiSessionPicker: 1, PromptEditor: 1, SaveAsModal: 3, WorkspaceSurface: 3,
    })
  })

  // Before the split branch contexts and the memoized TabContent: a switch rendered { a: 2, b: 2, c: 1 }, a save { a: 1, b: 1, c: 1 }
  // The one sidebar renders once for the tab in front. While the tab in front pushed its chat into the branch context,
  // a switch changed that context once ({ sidebar: 2, branches: 1 }), twice when a question was revealed. While a failed
  // quiet read retried by turning off for one commit, the real sidebar committed 3 times per switch
  it("a switch renders the two tabs it swaps, nothing else", async () => {
    expect(await count(() => fireEvent.keyDown(document.body, { key: "Tab" }))).toEqual({ a: 1, b: 1, c: 0, sidebar: 1, branches: 0 })
  })

  // The sidebar shows the refreshed history list
  it("a save renders no tab", async () => {
    branchMeta.current = { "a.json": { chat_id: "chat-a", parent_id: null } }
    expect(await count(() => fireEvent.keyDown(window, { key: "s", altKey: true }))).toEqual({ a: 0, b: 0, c: 0, sidebar: 1, branches: 1 })
    expect(saved).toHaveBeenCalled()
  })
})

/** The Save As dialog belongs to the tab that opened it; the prompt editor is one window over every tab. */
describe("dialogs across a tab switch", () => {
  beforeEach(async () => {
    const untitled = (id: string) => ({ ...tab(id), historyFile: null })
    sessionStorage.setItem("gigachad:tabs", JSON.stringify({ tabs: [untitled("a"), untitled("b")], activeId: "a" }))
    render(<AppShell />)
    await flush()
  })
  afterEach(() => {
    cleanup()
    saved.mockReset()
    promptLists.mockClear()
  })

  const switchTab = async (back = false) => {
    fireEvent.keyDown(document.body, { key: "Tab", shiftKey: back })
    await flush()
  }
  const saveAs = async () => {
    fireEvent.keyDown(window, { key: "s", altKey: true })
    await flush()
  }
  const nameField = () => screen.getByRole<HTMLInputElement>("textbox", { name: "Save Chat" })
  const type = (value: string) => fireEvent.change(nameField(), { target: { value } })

  it("keeps the name typed in a tab's dialog when the tab returns", async () => {
    await saveAs()
    type("Notes")
    await switchTab()
    await switchTab(true)
    expect(nameField().value).toBe("Notes")
  })

  it("leaves a tab's dialog open when another tab opens its own", async () => {
    await saveAs()
    type("Notes")
    await switchTab()
    await saveAs()
    type("Other")
    await switchTab(true)
    expect(nameField().value).toBe("Notes")
  })

  it("leaves another tab's dialog open when a tab's save resolves", async () => {
    let release = () => {}
    saved.mockImplementationOnce(() => new Promise<void>((resolve) => { release = resolve }))
    await saveAs()
    type("Notes")
    fireEvent.submit(nameField().form!)
    await switchTab()
    await saveAs()
    type("Other")
    await act(async () => release())
    // Past the dialog's exit animation
    await act(() => new Promise((resolve) => setTimeout(resolve, 300)))
    expect(nameField().value).toBe("Other")
  })

  it("keeps the prompt editor's draft over the other tab without reading its lists again", async () => {
    fireEvent.click(screen.getByText("Edit prompts"))
    fireEvent.click(await screen.findByText("Coder"))
    fireEvent.change(await screen.findByPlaceholderText("Write your prompt..."), { target: { value: "draft" } })
    await switchTab()
    expect(screen.getByPlaceholderText<HTMLTextAreaElement>("Write your prompt...").value).toBe("draft")
    expect(promptLists).toHaveBeenCalledTimes(1)
  })
})
