import { afterEach, describe, expect, it } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { Factory } from "lucide-react"
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

  it("shows `between` below the row whether open or not, outside the content it controls", () => {
    render(<Disclosure summary="Details" between={<p>Always</p>}><p>Inside</p></Disclosure>)
    const toggle = screen.getByRole("button", { name: "Details" })
    expect(screen.getByText("Always")).toBeTruthy()
    fireEvent.click(toggle)
    const content = document.getElementById(toggle.getAttribute("aria-controls") ?? "")
    expect(content).toContainElement(screen.getByText("Inside"))
    expect(content).not.toContainElement(screen.getByText("Always"))
    expect(toggle.compareDocumentPosition(screen.getByText("Always")) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it("draws an icon beside the summary, and a static row without content", () => {
    const { container } = render(<Disclosure summary="Notice" icon={Factory} />)
    expect(screen.queryByRole("button")).toBeNull()
    expect(screen.getByText("Notice")).toBeTruthy()
    expect(container.querySelectorAll("svg")).toHaveLength(1)
    cleanup()
    render(<Disclosure summary="Notice" icon={Factory}><p>Inside</p></Disclosure>)
    expect(screen.getByRole("button", { name: "Notice" }).querySelectorAll("svg")).toHaveLength(2)
  })
})
