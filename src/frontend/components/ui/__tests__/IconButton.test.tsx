import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { Play } from "lucide-react"
import { IconButton } from "../IconButton"

afterEach(cleanup)

describe("IconButton", () => {
  it("is a button named by its label that reports clicks", () => {
    const onClick = vi.fn()
    render(<IconButton icon={Play} label="Run cell" onClick={onClick} />)
    fireEvent.click(screen.getByRole("button", { name: "Run cell" }))
    expect(onClick).toHaveBeenCalledOnce()
  })
})
