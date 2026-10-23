import { StrictMode, useState } from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { StartupReadProvider, useStartupRead } from "../StartupReadContext"

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

function Resource({ label, load }: { label: string; load: (signal: AbortSignal) => Promise<string> }) {
  const [value, setValue] = useState("No data")
  const { loading, error, retry } = useStartupRead(label, load, setValue)
  return <section aria-label={label}>
    <output>{value}</output>
    <span>{loading ? "Loading" : error ? "Unavailable" : "Ready"}</span>
    <button onClick={retry}>Refresh {label}</button>
  </section>
}

function Workspace({ load, healthy }: { load: (signal: AbortSignal) => Promise<string>; healthy: (signal: AbortSignal) => Promise<string> }) {
  const [draft, setDraft] = useState("")
  return <StartupReadProvider>
    <textarea aria-label="Draft" value={draft} onChange={(event) => setDraft(event.target.value)} />
    <Resource label="Models" load={load} />
    <Resource label="Projects" load={healthy} />
  </StartupReadProvider>
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("startup read recovery", () => {
  it("retries only failed data and preserves the draft and successful content", async () => {
    const load = vi.fn<(signal: AbortSignal) => Promise<string>>()
      .mockRejectedValueOnce(new Error("Network unavailable"))
      .mockResolvedValueOnce("Recovered models")
    const healthy = vi.fn(async () => "Existing projects")
    render(<Workspace load={load} healthy={healthy} />)
    await screen.findByText("Network unavailable")
    fireEvent.change(screen.getByLabelText("Draft"), { target: { value: "Keep my unsent message" } })
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await screen.findByText("Recovered models")
    expect(screen.queryByLabelText("Startup connection problems")).not.toBeInTheDocument()
    expect(screen.getByLabelText("Draft")).toHaveValue("Keep my unsent message")
    expect(screen.getByText("Existing projects")).toBeVisible()
    expect(healthy).toHaveBeenCalledTimes(1)
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("times out a stalled read and rejects its late response after recovery", async () => {
    vi.useFakeTimers()
    const late = deferred<string>()
    let signal!: AbortSignal
    const load = vi.fn<(signal: AbortSignal) => Promise<string>>()
      .mockImplementationOnce((nextSignal) => { signal = nextSignal; return late.promise })
      .mockResolvedValueOnce("Recovered")
    render(<StartupReadProvider><Resource label="Models" load={load} /></StartupReadProvider>)
    await act(async () => { await vi.advanceTimersByTimeAsync(15_000) })
    expect(signal.aborted).toBe(true)
    expect(screen.getByText("Unavailable")).toBeVisible()
    expect(screen.getByText(/did not respond within/)).toBeVisible()
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Retry" })) })
    expect(screen.getByText("Recovered")).toBeVisible()
    await act(async () => { late.resolve("Obsolete") })
    expect(screen.queryByText("Obsolete")).not.toBeInTheDocument()
    expect(screen.getByText("Recovered")).toBeVisible()
  })

  it("keeps last good content on a later failure and does not retry automatically", async () => {
    const load = vi.fn<(signal: AbortSignal) => Promise<string>>()
      .mockResolvedValueOnce("Last good models")
      .mockRejectedValueOnce(new Error("Disconnected"))
    render(<StartupReadProvider><Resource label="Models" load={load} /></StartupReadProvider>)
    await screen.findByText("Last good models")
    fireEvent.click(screen.getByRole("button", { name: "Refresh Models" }))
    await screen.findByText("Disconnected")
    expect(screen.getByText("Last good models")).toBeVisible()
    expect(screen.getByText("Unavailable")).toBeVisible()
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("ignores obsolete navigation responses and removes unmounted failures", async () => {
    const stale = deferred<string>()
    let oldSignal!: AbortSignal
    const oldLoad = (signal: AbortSignal) => { oldSignal = signal; return stale.promise }
    const newLoad = async () => "Current project"
    const view = render(<StartupReadProvider><Resource label="Projects" load={oldLoad} /></StartupReadProvider>)
    view.rerender(<StartupReadProvider><Resource label="Projects" load={newLoad} /></StartupReadProvider>)
    await screen.findByText("Current project")
    expect(oldSignal.aborted).toBe(true)
    await act(async () => { stale.resolve("Old project") })
    expect(screen.queryByText("Old project")).not.toBeInTheDocument()
    const failing = async () => { throw new Error("Unavailable project") }
    view.rerender(<StartupReadProvider><Resource label="Projects" load={failing} /></StartupReadProvider>)
    await screen.findByText("Unavailable project")
    view.rerender(<StartupReadProvider><div>Closed workspace</div></StartupReadProvider>)
    await waitFor(() => expect(screen.queryByLabelText("Startup connection problems")).not.toBeInTheDocument())
  })

  it("aborts the discarded StrictMode request without surfacing a false error", async () => {
    const first = deferred<string>()
    const signals: AbortSignal[] = []
    const load = vi.fn((signal: AbortSignal) => {
      signals.push(signal)
      return signals.length === 1 ? first.promise : Promise.resolve("Ready models")
    })
    render(<StrictMode><StartupReadProvider><Resource label="Models" load={load} /></StartupReadProvider></StrictMode>)
    await screen.findByText("Ready models")
    expect(signals[0].aborted).toBe(true)
    await act(async () => { first.resolve("Discarded") })
    expect(screen.queryByText("Discarded")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Startup connection problems")).not.toBeInTheDocument()
  })
})
