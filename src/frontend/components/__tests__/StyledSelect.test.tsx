import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { StyledSelect } from "@/components/StyledSelect"

describe("StyledSelect", () => {
  it("opens an app-styled listbox and selects an option", () => {
    const onChange = vi.fn()
    render(<StyledSelect ariaLabel="Model" value="fast" onChange={onChange} options={[{ value: "fast", label: "Fast" }, { value: "smart", label: "Smart" }]} />)

    fireEvent.click(screen.getByRole("button", { name: "Model" }))
    fireEvent.click(screen.getByRole("option", { name: "Smart" }))

    expect(onChange).toHaveBeenCalledWith("smart")
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument()
  })

  it("closes with Escape without changing the value", () => {
    const onChange = vi.fn()
    render(<StyledSelect value="fast" onChange={onChange} options={[{ value: "fast", label: "Fast" }]} />)

    const trigger = screen.getByRole("button", { name: "Select option" })
    fireEvent.click(trigger)
    fireEvent.keyDown(trigger, { key: "Escape" })

    expect(screen.queryByRole("listbox")).not.toBeInTheDocument()
    expect(onChange).not.toHaveBeenCalled()
  })
})
