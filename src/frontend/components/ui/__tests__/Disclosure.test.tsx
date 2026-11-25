import { afterEach, describe, expect, it } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { Disclosure } from "../Disclosure"

afterEach(cleanup)

describe("Disclosure", () => {
  it("is a button named by its summary that mounts its content only while open", () => {
    render(<Disclosure summary="Details"><p>Inside</p></Disclosure>)
    const toggle = screen.getByRole("button", { name: "Details" })
    expect(toggle).toHaveAttribute("aria-expanded", "false")
    expect(toggle).not.toHaveAttribute("aria-controls")
    expect(screen.queryByText("Inside")).toBeNull()

    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute("aria-expanded", "true")
    const content = screen.getByText("Inside").parentElement
    expect(toggle).toHaveAttribute("aria-controls", content?.id)

    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByText("Inside")).toBeNull()
  })

  it("starts open with defaultOpen", () => {
    render(<Disclosure summary="Details" defaultOpen><p>Inside</p></Disclosure>)
    expect(screen.getByRole("button", { name: "Details" })).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByText("Inside")).toBeTruthy()
  })
})
