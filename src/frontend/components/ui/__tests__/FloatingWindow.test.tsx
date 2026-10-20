import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { FloatingWindow } from "../FloatingWindow"

describe("FloatingWindow backdrop", () => {
  it("closes on a click outside the panel", () => {
    const onClose = vi.fn()
    render(<FloatingWindow onClose={onClose} label="w">body</FloatingWindow>)
    fireEvent.click(screen.getByRole("dialog").parentElement!)
    expect(onClose).toHaveBeenCalledOnce()
  })

  it("ignores that click when dismissOnBackdrop is false but still closes on Escape", () => {
    const onClose = vi.fn()
    render(<FloatingWindow onClose={onClose} label="w" dismissOnBackdrop={false}>body</FloatingWindow>)
    fireEvent.click(screen.getByRole("dialog").parentElement!)
    expect(onClose).not.toHaveBeenCalled()
    fireEvent.keyDown(window, { key: "Escape" })
    expect(onClose).toHaveBeenCalledOnce()
  })
})
