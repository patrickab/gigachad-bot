import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { StartupReadProvider } from "@/contexts/StartupReadContext"
import { OmpProvider, useOmp } from "../OmpProvider"

const settings = vi.hoisted(() => ({ hostAccess: false, ompWebUrl: null as string | null }))
const fetchOmpOverview = vi.hoisted(() => vi.fn(async () => {
  throw new Error("OMP access refused")
}))
vi.mock("@/contexts/SettingsContext", () => ({ useSettings: () => settings }))
vi.mock("@/lib/backend/api", () => ({ fetchOmpOverview }))
vi.mock("@/features/editor/FileHoverView", () => ({ useOpenFileHoverView: () => vi.fn() }))
vi.mock("@/hooks/useThemeName", () => ({ useThemeName: () => "light" }))

function Workspace() {
  const { refreshOverview } = useOmp()
  return <button onClick={refreshOverview}>Refresh OMP</button>
}

function App() {
  return <StartupReadProvider><OmpProvider><Workspace /></OmpProvider></StartupReadProvider>
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  fetchOmpOverview.mockClear()
  settings.hostAccess = false
})

it("does not read or poll host-only OMP data without access, including explicit refresh", async () => {
  vi.useFakeTimers()
  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "Refresh OMP" }))
  fireEvent(window, new Event("focus"))
  await act(async () => { await vi.advanceTimersByTimeAsync(30_000) })
  expect(fetchOmpOverview).not.toHaveBeenCalled()
  expect(screen.queryByLabelText("Startup connection problems")).not.toBeInTheDocument()
})

it("starts recovery after access is granted and removes warnings and polling when revoked", async () => {
  vi.useFakeTimers()
  const view = render(<App />)
  settings.hostAccess = true
  await act(async () => { view.rerender(<App />) })
  expect(screen.getByText("OMP access refused")).toBeInTheDocument()
  expect(fetchOmpOverview).toHaveBeenCalledTimes(1)

  settings.hostAccess = false
  await act(async () => { view.rerender(<App />) })
  fireEvent.click(screen.getByRole("button", { name: "Refresh OMP" }))
  fireEvent(window, new Event("focus"))
  await act(async () => { await vi.advanceTimersByTimeAsync(30_000) })
  expect(fetchOmpOverview).toHaveBeenCalledTimes(1)
  expect(screen.queryByLabelText("Startup connection problems")).not.toBeInTheDocument()
})
