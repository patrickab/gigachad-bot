import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, waitFor } from "@testing-library/react"
import { ProviderUsage, formatReset } from "@/features/navigation/ProviderUsage"
import type { ProviderUsageReport } from "@/lib/domain/types"

const api = vi.hoisted(() => ({ fetchProviderUsage: vi.fn<(refresh?: boolean) => Promise<{ reports: ProviderUsageReport[] }>>() }))
vi.mock("@/lib/backend/api", () => api)

const reports: ProviderUsageReport[] = [
  { provider: "openai-codex", account: "me@example.com", no_limits: false, windows: [
    { id: "5h", percent: 0, resets_in_minutes: 296 }, { id: "7d", percent: 28.4, resets_in_minutes: 9084 },
  ] },
  { provider: "anthropic", account: null, no_limits: false, windows: [{ id: "5h", percent: 91, resets_in_minutes: 30 }] },
  { provider: "ollama", account: null, no_limits: true, windows: [] },
]

beforeEach(() => {
  const stored = new Map<string, string>()
  vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => { stored.set(key, value) } })
  api.fetchProviderUsage.mockReset().mockResolvedValue({ reports })
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe("ProviderUsage", () => {
  it("summarises the lowest five-hour usage and reveals other windows on demand", async () => {
    const view = render(<ProviderUsage />)

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
    const view = render(<ProviderUsage />)
    await view.findByRole("button", { name: /Usage.*5-hour usage: 0%/ })

    api.fetchProviderUsage.mockRejectedValueOnce(new Error("502"))
    fireEvent.click(view.getByRole("button", { name: "Refresh usage" }))

    expect(api.fetchProviderUsage).toHaveBeenLastCalledWith(true)
    expect(await view.findByText("unavailable")).toBeVisible()
    await waitFor(() => expect(view.getByRole("button", { name: "Refresh usage" })).toBeEnabled())
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
    const view = render(<ProviderUsage />)
    expect(await view.findByText("unavailable")).toBeVisible()
    expect(view.getByText("anthropic")).toBeVisible()
  })
})
