import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { ViewSwitch } from "@/components/ui/SegmentedSwitch"

describe("ViewSwitch", () => {
  it("renders the shared segmented look by default", () => {
    render(<ViewSwitch value="a" onChange={() => {}} options={[["a", "A"], ["b", "B"]]} />)
    expect(screen.getByRole("group").className).toBe("flex gap-px rounded-md bg-surface p-0.5")
    expect(screen.getByRole("button", { name: "A" }).className).toBe("rounded px-2 py-0.5 text-xs transition-colors bg-surface-elevated text-ink")
    expect(screen.getByRole("button", { name: "B" }).className).toBe("rounded px-2 py-0.5 text-xs transition-colors text-ink-subtle hover:text-ink")
  })
})
