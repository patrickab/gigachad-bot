import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { UndoDeleteProvider, useUndoDelete } from "../UndoDeleteContext"

function Controls({ run }: { run: (key: string) => Promise<void> }) {
  const { schedule, pending } = useUndoDelete()
  return <>
    <button onClick={() => schedule("one", "first", () => run("one"))}>Delete first</button>
    <button onClick={() => schedule("two", "second", () => run("two"))}>Delete second</button>
    <span>{pending("one") ? "first hidden" : "first visible"}</span>
    <span>{pending("two") ? "second hidden" : "second visible"}</span>
  </>
}

const mount = (run: (key: string) => Promise<void>) => render(<UndoDeleteProvider><Controls run={run} /></UndoDeleteProvider>)
afterEach(() => vi.useRealTimers())

it("undo restores a deleted item without sending a delete", async () => {
  vi.useFakeTimers()
  const run = vi.fn(async () => {})
  mount(run)
  fireEvent.click(screen.getByText("Delete first"))
  expect(screen.getByText("first hidden")).toBeTruthy()
  fireEvent.click(screen.getByRole("button", { name: "Undo" }))
  await act(async () => { vi.advanceTimersByTime(7000) })
  expect(screen.getByText("first visible")).toBeTruthy()
  expect(run).not.toHaveBeenCalled()
})

it("commits after the undo window and restores on a server refusal", async () => {
  vi.useFakeTimers()
  const run = vi.fn(async () => { throw new Error("Referenced by another file") })
  mount(run)
  fireEvent.click(screen.getByText("Delete first"))
  await act(async () => { vi.advanceTimersByTime(6000) })
  expect(run).toHaveBeenCalledWith("one")
  expect(screen.getByText("first visible")).toBeTruthy()
  expect(screen.getByText("Referenced by another file")).toBeTruthy()
})

it("a second delete commits the first before starting its own undo window", async () => {
  vi.useFakeTimers()
  const run = vi.fn(async () => {})
  mount(run)
  fireEvent.click(screen.getByText("Delete first"))
  fireEvent.click(screen.getByText("Delete second"))
  await act(async () => {})
  expect(run).toHaveBeenCalledWith("one")
  expect(screen.getByText("second hidden")).toBeTruthy()
  fireEvent.click(screen.getByRole("button", { name: "Undo" }))
  expect(run).toHaveBeenCalledTimes(1)
})

it("keeps an earlier deletion hidden while a second item is undoable", async () => {
  vi.useFakeTimers()
  let finishFirst!: () => void
  const run = vi.fn((key: string) => key === "one"
    ? new Promise<void>((resolve) => { finishFirst = resolve })
    : Promise.resolve())
  mount(run)
  fireEvent.click(screen.getByText("Delete first"))
  fireEvent.click(screen.getByText("Delete second"))
  await act(async () => {})
  expect(screen.getByText("first hidden")).toBeTruthy()
  expect(screen.getByText("second hidden")).toBeTruthy()
  fireEvent.click(screen.getByRole("button", { name: "Undo" }))
  expect(screen.getByText("first hidden")).toBeTruthy()
  expect(screen.getByText("second visible")).toBeTruthy()
  await act(async () => { finishFirst() })
  expect(screen.getByText("first visible")).toBeTruthy()
})

function Optimistic({ run, revert }: { run: () => Promise<void>, revert: () => void }) {
  const { schedule } = useUndoDelete()
  return <button onClick={() => schedule("opt", "item", run, revert)}>Delete item</button>
}
const mountOptimistic = (run: () => Promise<void>, revert: () => void) =>
  render(<UndoDeleteProvider><Optimistic run={run} revert={revert} /></UndoDeleteProvider>)

it("reverts an optimistic change on undo and on a refused commit, but not on success", async () => {
  vi.useFakeTimers()
  const revert = vi.fn()
  const ok = mountOptimistic(async () => {}, revert)
  fireEvent.click(screen.getByText("Delete item"))
  fireEvent.click(screen.getByRole("button", { name: "Undo" }))
  expect(revert).toHaveBeenCalledTimes(1)
  fireEvent.click(screen.getByText("Delete item"))
  await act(async () => { vi.advanceTimersByTime(6000) })
  expect(revert).toHaveBeenCalledTimes(1)
  ok.unmount()

  mountOptimistic(async () => { throw new Error("refused") }, revert)
  fireEvent.click(screen.getByText("Delete item"))
  await act(async () => { vi.advanceTimersByTime(6000) })
  expect(revert).toHaveBeenCalledTimes(2)
})
