import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { EnabledToolsProvider, useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { TOOLS } from "@/features/chat/tools"

const login = vi.hoisted(() => ({ hostAccess: false }))
vi.mock("@/contexts/SettingsContext", () => ({ useHostAccess: () => login.hostAccess }))

beforeEach(() => {
  const stored = new Map<string, string>()
  vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => { stored.set(key, value) } })
})

afterEach(() => {
  login.hostAccess = false
  vi.unstubAllGlobals()
})

function renderTools() {
  return renderHook(() => useEnabledTools(), { wrapper: EnabledToolsProvider })
}

describe("useEnabledTools", () => {
  it("offers the default tools and toggles one without disturbing the others", () => {
    const { result } = renderHook(() => useEnabledTools(), { wrapper: EnabledToolsProvider })

    expect(result.current.enabledTools).toEqual(TOOLS.filter((tool) => tool.defaultEnabled && !tool.hostOnly).map((tool) => tool.name))
    expect(result.current.enabledTools).not.toContain("workspace_agent")

    act(() => result.current.toggleTool("sandbox_plot"))

    expect(result.current.enabledTools).not.toContain("sandbox_plot")
    expect(result.current.enabledTools).toContain("web_search")

    act(() => result.current.toggleTool("workspace_agent"))

    expect(result.current.enabledTools).toContain("workspace_agent")
  })

  it("offers learn sessions to host logins only", () => {
    const guest = renderHook(() => useEnabledTools(), { wrapper: EnabledToolsProvider })
    expect(guest.result.current.enabledTools).not.toContain("start_learn_session")
    guest.unmount()

    login.hostAccess = true
    const host = renderHook(() => useEnabledTools(), { wrapper: EnabledToolsProvider })
    expect(host.result.current.enabledTools).toContain("start_learn_session")
  })

  it("enables a new default tool in a list saved before it existed, once", () => {
    login.hostAccess = true
    localStorage.setItem("gigachad-enabled-tools", JSON.stringify(["web_search"]))

    const before = renderTools()
    expect(before.result.current.enabledTools).toEqual(["web_search", "start_learn_session"])
    act(() => before.result.current.toggleTool("start_learn_session"))
    expect(before.result.current.enabledTools).toEqual(["web_search"])
    before.unmount()

    const after = renderTools()
    expect(after.result.current.enabledTools).toEqual(["web_search"])
  })

  it("keeps an opted-out default tool off across reloads", () => {
    const first = renderTools()
    act(() => first.result.current.toggleTool("web_search"))
    first.unmount()

    const second = renderTools()
    expect(second.result.current.enabledTools).not.toContain("web_search")
    expect(second.result.current.enabledTools).toContain("deep_research")
  })
})
