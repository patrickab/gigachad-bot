import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, waitFor } from "@testing-library/react"
import { ProviderUsage, ProviderUsagePoll, formatReset } from "@/features/navigation/ProviderUsage"
import type { ProviderUsageReport } from "@/lib/domain/types"

const api = vi.hoisted(() => ({ fetchProviderUsage: vi.fn<(refresh?: boolean) => Promise<{ reports: ProviderUsageReport[] }>>() }))
vi.mock("@/lib/backend/api", () => api)
const sidebarFrame = vi.hoisted(() => ({ collapsed: false, toggleCollapsed: () => {} }))
vi.mock("@/features/navigation/SidebarContext", () => ({ useSidebar: () => sidebarFrame }))

const reports: ProviderUsageReport[] = [
  { provider: "openai-codex", account: "me@example.com", no_limits: false, windows: [
    { id: "5h", percent: 0, resets_in_minutes: 296 }, { id: "7d", percent: 28.4, resets_in_minutes: 9084 },
  ] },
  { provider: "anthropic", account: null, no_limits: false, windows: [{ id: "5h", percent: 91, resets_in_minutes: 30 }] },
  { provider: "ollama", account: null, no_limits: true, windows: [] },
]

/** The app shell's poll beside `sidebars` usage sections, one per tab. */
const renderApp = (sidebars = 1) => render(<><ProviderUsagePoll />{Array.from({ length: sidebars }, (_, index) => <ProviderUsage key={index} />)}</>)

beforeEach(() => {
  const stored = new Map<string, string>()
  vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => { stored.set(key, value) } })
  api.fetchProviderUsage.mockReset().mockResolvedValue({ reports })
  sidebarFrame.collapsed = false
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe("ProviderUsage", () => {
  it("summarises the lowest five-hour usage and reveals other windows on demand", async () => {
    const view = renderApp()

    expect(await view.findByRole("button", { name: /Usage.*5-hour usage: 0%/ })).toBeVisible()
    expect(view.queryByText("openai-codex")).not.toBeInTheDocument()

    fireEvent.click(view.getByText("Usage"))
    expect(view.getByText("openai-codex")).toBeVisible()
    expect(view.queryByText("28%")).not.toBeInTheDocument()
    expect(view.queryByText("ollama")).not.toBeInTheDocument()

    fireEvent.click(view.getByRole("button", { name: "openai-codex" }))
    expect(view.getByText("in 6d 7h")).toBeVisible()
    expect(view.getByText("28%")).toBeVisible()
  })

  it("refreshes past the server cache and says so when usage cannot be read", async () => {
    const view = renderApp()
    await view.findByRole("button", { name: /Usage.*5-hour usage: 0%/ })

    api.fetchProviderUsage.mockRejectedValueOnce(new Error("502"))
    fireEvent.click(view.getByRole("button", { name: "Refresh usage" }))

    expect(api.fetchProviderUsage).toHaveBeenLastCalledWith(true)
    expect(await view.findByText("unavailable")).toBeVisible()
    await waitFor(() => expect(view.getByRole("button", { name: "Refresh usage" })).toBeEnabled())
  })

  it("reads once for every tab, nothing for a tab shown later, and shares a refresh and its failure with every tab", async () => {
    const view = renderApp(2)
    await waitFor(() => expect(view.getAllByRole("button", { name: /Usage.*5-hour usage: 0%/ })).toHaveLength(2))
    expect(api.fetchProviderUsage).toHaveBeenCalledOnce()

    view.rerender(<><ProviderUsagePoll /><ProviderUsage key={0} /><ProviderUsage key={1} /><ProviderUsage key={2} /></>)
    expect(view.getAllByRole("button", { name: /Usage.*5-hour usage: 0%/ })).toHaveLength(3)
    expect(api.fetchProviderUsage).toHaveBeenCalledOnce()

    api.fetchProviderUsage.mockRejectedValueOnce(new Error("502"))
    fireEvent.click(view.getAllByRole("button", { name: "Refresh usage" })[0])
    await waitFor(() => expect(view.getAllByText("unavailable")).toHaveLength(3))

    api.fetchProviderUsage.mockResolvedValueOnce({ reports: [{ provider: "anthropic", account: null, no_limits: false, windows: [{ id: "5h", percent: 40, resets_in_minutes: 30 }] }] })
    fireEvent.click(view.getAllByRole("button", { name: "Refresh usage" })[1])
    await waitFor(() => expect(view.getAllByRole("button", { name: /Usage.*5-hour usage: 40%/ })).toHaveLength(3))
  })

  it("reads nothing while the sidebar is closed and reads at once when it opens", async () => {
    sidebarFrame.collapsed = true
    const view = render(<ProviderUsagePoll />)
    expect(api.fetchProviderUsage).not.toHaveBeenCalled()

    sidebarFrame.collapsed = false
    view.rerender(<ProviderUsagePoll />)
    await waitFor(() => expect(api.fetchProviderUsage).toHaveBeenCalledOnce())
  })

  it("formats reset times", () => {
    expect([45, 60, 296, 1440, 9084].map(formatReset)).toEqual(["45m", "1h", "4h 56m", "1d", "6d 7h"])
  })
})

describe("ProviderUsage unavailable accounts", () => {
  it("keeps an account omp could not read as an unavailable row", async () => {
    api.fetchProviderUsage.mockResolvedValue({ reports: [
      ...reports.slice(0, 1),
      { provider: "anthropic", account: "me@example.com", no_limits: false, unavailable: true, windows: [] },
    ] })
    localStorage.setItem("provider_usage_open", "true")
    const view = renderApp()
    expect(await view.findByText("unavailable")).toBeVisible()
    expect(view.getByText("anthropic")).toBeVisible()
  })
})
