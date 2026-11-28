import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import type { ReactNode } from "react"
import { TabContext, type Tab } from "@/contexts/TabContext"
import type { ChatListing, ChatSearchHit } from "@/lib/backend/api"
import { ChatPicker } from "../ChatPicker"

const api = vi.hoisted(() => ({
  chats: [] as ChatListing[],
  search: vi.fn<(query: string, limit?: number, signal?: AbortSignal) => Promise<ChatSearchHit[]>>(),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  listAllChats: async () => api.chats,
  searchChats: (query: string, limit?: number, signal?: AbortSignal) => api.search(query, limit, signal),
}))
vi.mock("@/contexts/ProjectContext", () => ({
  useProject: () => ({ projects: [{ name: "Thesis", slug: "thesis" }, { name: "Garden", slug: "garden" }] }),
}))

const HOUR = 60 * 60 * 1000

function chat(historyFile: string, title: string, ageHours: number, projectSlug: string | null = null): ChatListing {
  return { historyFile, projectSlug, title, updatedAt: new Date(Date.now() - ageHours * HOUR).toISOString() }
}

const TAB: Tab = { id: "tab", chatId: "chat", historyFile: null, title: null, config: {} as Tab["config"], projectSlug: null, workspaceItem: null, itemHidden: false }

function inTab(tab: Tab, isActive: boolean, children: ReactNode) {
  return <TabContext.Provider value={{ tab, isActive, setModeLabel: vi.fn(), updateConfig: vi.fn(), hideItem: vi.fn(), dropItem: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
    {children}
  </TabContext.Provider>
}

function renderPicker({ isActive = true, open = true } = {}) {
  const onOpenChat = vi.fn()
  render(inTab(TAB, isActive, <ChatPicker onOpenChat={onOpenChat} />))
  if (open) fireEvent.keyDown(window, { key: "c", altKey: true })
  return onOpenChat
}

const search = () => screen.getByRole("combobox", { name: "Search chats…" })
const titles = (group: HTMLElement) => within(group).getAllByRole("option").map((option) => option.firstElementChild?.firstElementChild?.textContent)

afterEach(() => {
  cleanup()
  api.chats = []
  api.search.mockReset()
})

describe("ChatPicker", () => {
  it("opens on Alt+C with the chats grouped by project, groups by their newest chat, chats newest first, General for those without", async () => {
    api.chats = [
      chat("thesis/a.json", "Thesis old", 5, "thesis"),
      chat("loose.json", "Loose", 2),
      chat("garden/b.json", "Garden", 3, "garden"),
      chat("thesis/c.json", "Thesis new", 1, "thesis"),
      chat("other.json", "Other", 4),
    ]
    renderPicker()

    const groups = await screen.findAllByRole("group")
    expect(groups.map((group) => document.getElementById(group.getAttribute("aria-labelledby")!)?.textContent)).toEqual(["Thesis", "General", "Garden"])
    expect(titles(groups[0])).toEqual(["Thesis new", "Thesis old"])
    expect(titles(groups[1])).toEqual(["Loose", "Other"])
  })

  it("opens a picked chat without a pair to reveal", async () => {
    api.chats = [chat("loose.json", "Loose", 2)]
    const onOpenChat = renderPicker()

    fireEvent.click(await screen.findByRole("option", { name: /Loose/ }))
    expect(onOpenChat).toHaveBeenCalledWith("loose.json")
    expect(screen.queryByRole("dialog", { name: "Chats" })).not.toBeInTheDocument()
  })

  it("lists title matches and every content hit as a row, the first naming the chat, the match bold, and opens a hit at its pair", async () => {
    api.chats = [chat("titled.json", "Refactor tabs", 1), chat("other.json", "Unrelated", 2)]
    api.search.mockResolvedValue([
      { ...chat("titled.json", "Refactor tabs", 1), matches: [{ qaIndex: 0, role: "user", ordinal: 0, snippet: "refactor me" }] },
      { ...chat("other.json", "Unrelated", 2), matches: [
        { qaIndex: 1, role: "user", ordinal: 0, snippet: "…we should Refactor the store…" },
        { qaIndex: 3, role: "assistant", ordinal: 0, snippet: "and refactoring later" },
      ] },
    ])
    const onOpenChat = renderPicker()
    await screen.findByRole("option", { name: /Refactor tabs/ })

    fireEvent.change(search(), { target: { value: " refactor " } })
    const content = await screen.findByRole("group", { name: "Content" })
    expect(api.search).toHaveBeenCalledWith("refactor", 20, expect.any(AbortSignal))
    expect(within(screen.getByRole("group", { name: "Chats" })).getAllByRole("option").map((row) => row.textContent)).toEqual(["Refactor tabs1h", "refactor me"])
    const rows = within(content).getAllByRole("option")
    expect(rows).toHaveLength(2)
    expect(rows[0]).toHaveTextContent("Unrelated…we should Refactor the store…")
    expect(within(rows[0]).getByText("Refactor").tagName).toBe("STRONG")
    expect(rows[1]).toHaveTextContent("and refactoring later")
    expect(rows[1]).not.toHaveTextContent("Unrelated")
    expect(within(rows[1]).getByText("refactor").tagName).toBe("STRONG")

    fireEvent.click(rows[1])
    expect(onOpenChat).toHaveBeenCalledWith("other.json", 3, { query: "refactor", role: "assistant", ordinal: 0 })
  })

  it("lists the hits of a chat whose title matches after its title row and reveals the hit with its query", async () => {
    api.chats = [chat("titled.json", "Refactor tabs", 1)]
    api.search.mockResolvedValue([
      { ...chat("titled.json", "Refactor tabs", 1), matches: [{ qaIndex: 2, role: "user", ordinal: 0, snippet: "refactor me" }] },
    ])
    const onOpenChat = renderPicker()
    await screen.findByRole("option", { name: /Refactor tabs/ })

    fireEvent.change(search(), { target: { value: "refactor" } })
    await waitFor(() => expect(within(screen.getByRole("group", { name: "Chats" })).getAllByRole("option")).toHaveLength(2))
    const rows = within(screen.getByRole("group", { name: "Chats" })).getAllByRole("option")
    expect(rows[0]).toHaveTextContent("Refactor tabs")
    fireEvent.click(rows[1])
    expect(onOpenChat).toHaveBeenCalledWith("titled.json", 2, { query: "refactor", role: "user", ordinal: 0 })
  })

  it("hands a query longer than 200 characters to the reveal cut as it was searched", async () => {
    api.chats = []
    const long = "a".repeat(250)
    api.search.mockResolvedValue([{ ...chat("x.json", "X", 1), matches: [{ qaIndex: 1, role: "user", ordinal: 0, snippet: long }] }])
    const onOpenChat = renderPicker()
    fireEvent.change(search(), { target: { value: long } })
    fireEvent.click(await screen.findByRole("option", { name: /X/ }))
    expect(api.search).toHaveBeenCalledWith("a".repeat(200), 20, expect.any(AbortSignal))
    expect(onOpenChat).toHaveBeenCalledWith("x.json", 1, { query: "a".repeat(200), role: "user", ordinal: 0 })
  })

  it("aborts the request of a superseded query", async () => {
    api.search.mockImplementation(() => new Promise(() => {}))
    renderPicker()

    fireEvent.change(search(), { target: { value: "ab" } })
    await waitFor(() => expect(api.search).toHaveBeenCalledTimes(1))
    const first = api.search.mock.calls[0][2]!
    expect(first.aborted).toBe(false)

    fireEvent.change(search(), { target: { value: "abc" } })
    expect(first.aborted).toBe(true)
    await waitFor(() => expect(api.search).toHaveBeenCalledWith("abc", 20, expect.any(AbortSignal)))
    expect(api.search).toHaveBeenCalledTimes(2)
  })

  it("closes on Escape", async () => {
    renderPicker()
    fireEvent.keyDown(search(), { key: "Escape" })
    expect(screen.queryByRole("dialog", { name: "Chats" })).not.toBeInTheDocument()
  })

  it("closes when another tab comes to the front of the app's chrome", () => {
    const picker = <ChatPicker onOpenChat={vi.fn()} />
    const view = render(inTab(TAB, true, picker))
    fireEvent.keyDown(window, { key: "c", altKey: true })
    expect(screen.getByRole("dialog", { name: "Chats" })).toBeInTheDocument()

    view.rerender(inTab({ ...TAB, id: "other" }, true, picker))
    expect(screen.queryByRole("dialog", { name: "Chats" })).not.toBeInTheDocument()
  })

  it("ignores Alt+C with another modifier, from a terminal, and in a background tab", () => {
    renderPicker({ open: false })
    fireEvent.keyDown(window, { key: "c", altKey: true, ctrlKey: true })
    fireEvent.keyDown(window, { key: "c", altKey: true, shiftKey: true })
    const terminal = document.createElement("div")
    terminal.setAttribute("data-terminal", "")
    const input = document.createElement("textarea")
    terminal.appendChild(input)
    document.body.appendChild(terminal)
    fireEvent.keyDown(input, { key: "c", altKey: true })
    terminal.remove()
    expect(screen.queryByRole("dialog", { name: "Chats" })).not.toBeInTheDocument()
    cleanup()

    renderPicker({ isActive: false })
    expect(screen.queryByRole("dialog", { name: "Chats" })).not.toBeInTheDocument()
  })
})
