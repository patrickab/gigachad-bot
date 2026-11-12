import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { StartupReadProvider } from "@/contexts/StartupReadContext"
import { useResource, useResourceFamily, useResourceState } from "@/hooks/useResource"
import { createResource, resourceFamily } from "@/lib/resource"

const read = vi.fn<(signal: AbortSignal) => Promise<string>>()
const greeting = createResource(read)

/** Shows the shared greeting and stores its own on click, like a tab saving a setting. */
function Consumer({ name }: { name: string }) {
  const value = useResource("Greeting", greeting)
  return <button onClick={() => greeting.set(`saved by ${name}`)}>{`${name}: ${value ?? "loading"}`}</button>
}

function FailureView() {
  const { value, error } = useResourceState("Greeting", greeting)
  return <span>{value ?? error ?? "loading"}</span>
}

/** The greeting read quietly while `enabled`, under the real startup banner. */
function QuietView({ enabled }: { enabled: boolean }) {
  const { value, error } = useResourceState(null, greeting, enabled)
  return <span>{value ?? error ?? "loading"}</span>
}

const lengthRead = vi.fn(async (key: string) => key.length)
const lengths = resourceFamily(lengthRead)

describe("useResource", () => {
  it("loads once for every mounted consumer and shows one consumer's stored value in the other", async () => {
    const { promise, resolve } = Promise.withResolvers<string>()
    read.mockReset().mockReturnValue(promise)
    render(<><Consumer name="a" /><Consumer name="b" /></>)

    await act(async () => resolve("hello"))
    expect(screen.getByText("a: hello")).toBeVisible()
    expect(screen.getByText("b: hello")).toBeVisible()
    expect(read).toHaveBeenCalledOnce()

    fireEvent.click(screen.getByText("a: hello"))
    expect(screen.getByText("b: saved by a")).toBeVisible()
  })

  it("reads nothing for a consumer mounting after the value loaded", async () => {
    read.mockReset().mockResolvedValue("hello")
    const view = render(<Consumer name="a" />)
    expect(await view.findByText("a: hello")).toBeVisible()

    view.rerender(<><Consumer name="a" /><Consumer name="b" /></>)
    expect(screen.getByText("b: hello")).toBeVisible()
    expect(read).toHaveBeenCalledOnce()
  })

  it("reports why the first read failed until a value arrives", async () => {
    read.mockReset().mockRejectedValue(new Error("backend down"))
    render(<FailureView />)
    expect(await screen.findByText("backend down")).toBeVisible()

    act(() => greeting.set("hello"))
    expect(screen.getByText("hello")).toBeVisible()
  })

  it("reports the failure of the read that replaced a superseded first read", async () => {
    read.mockReset().mockReturnValueOnce(new Promise(() => {})).mockRejectedValueOnce(new Error("backend down"))
    render(<FailureView />)
    await waitFor(() => expect(read).toHaveBeenCalledOnce())

    act(() => greeting.invalidate())
    expect(await screen.findByText("backend down")).toBeVisible()
    expect(read).toHaveBeenCalledTimes(2)
  })

  it("keeps a quiet read's failure out of the banner and reads again once enabled again", async () => {
    read.mockReset().mockRejectedValueOnce(new Error("backend down")).mockResolvedValue("hello")
    const view = render(<StartupReadProvider><QuietView enabled /></StartupReadProvider>)
    await waitFor(() => expect(read).toHaveBeenCalledOnce())
    await act(async () => {})
    expect(screen.getByText("loading")).toBeVisible()
    expect(screen.queryByRole("region", { name: "Startup connection problems" })).not.toBeInTheDocument()

    view.rerender(<StartupReadProvider><QuietView enabled={false} /></StartupReadProvider>)
    view.rerender(<StartupReadProvider><QuietView enabled /></StartupReadProvider>)
    expect(await screen.findByText("hello")).toBeVisible()
    expect(read).toHaveBeenCalledTimes(2)
  })
})

describe("useResourceFamily", () => {
  it("loads each missing member once for every consumer and keeps its snapshot while no member changed", async () => {
    lengthRead.mockClear()
    const first = renderHook(({ keys }) => useResourceFamily("Lengths", lengths, keys), { initialProps: { keys: ["a", "bb"] } })
    const second = renderHook(() => useResourceFamily("Lengths", lengths, ["a", "bb"]))
    await waitFor(() => expect(first.result.current).toEqual({ a: 1, bb: 2 }))
    expect(second.result.current).toEqual({ a: 1, bb: 2 })
    expect(lengthRead).toHaveBeenCalledTimes(2)

    const loaded = first.result.current
    first.rerender({ keys: ["a", "bb"] })
    expect(first.result.current).toBe(loaded)

    act(() => lengths("bb").set(5))
    expect(first.result.current).toEqual({ a: 1, bb: 5 })

    first.rerender({ keys: ["a", "bb", "ccc"] })
    await waitFor(() => expect(first.result.current).toEqual({ a: 1, bb: 5, ccc: 3 }))
    expect(lengthRead).toHaveBeenCalledTimes(3)
  })
})
