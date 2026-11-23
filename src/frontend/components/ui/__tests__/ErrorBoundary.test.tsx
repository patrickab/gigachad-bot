import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { ErrorBoundary } from "../ErrorBoundary"

afterEach(cleanup)

let broken = true
function Bomb() {
  if (broken) throw new Error("boom")
  return <p>fine</p>
}

describe("ErrorBoundary", () => {
  it("shows the message, logs it, and recovers on reset; siblings stay", () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {})
    broken = true
    render(<><ErrorBoundary><Bomb /></ErrorBoundary><p>other</p></>)
    expect(screen.getByRole("alert").textContent).toContain("boom")
    expect(screen.getByText("other")).toBeTruthy()
    expect(log.mock.calls.some((call) => call[0] === "ErrorBoundary caught an error")).toBe(true)
    broken = false
    fireEvent.click(screen.getByRole("button", { name: "Reload tab" }))
    expect(screen.getByText("fine")).toBeTruthy()
    log.mockRestore()
  })
})
