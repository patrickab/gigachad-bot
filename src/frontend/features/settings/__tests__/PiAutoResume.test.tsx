import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SettingsMenu } from "@/features/settings/SettingsMenu"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"

const state = vi.hoisted(() => ({ value: false as boolean | null, set: vi.fn() }))

vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/contexts/SettingsContext")>()),
  usePiAutoResume: () => [state.value, state.set],
}))

function open() {
  render(<SettingsMenu prompts={{}} config={DEFAULT_TAB_CONFIG} onConfigChange={vi.fn()} models={null} researchEnabled={false} />)
  fireEvent.click(screen.getByRole("button", { name: "Settings" }))
}

describe("resume interrupted sessions setting", () => {
  it("switches pi-host's auto-resume on", () => {
    state.value = false
    open()

    fireEvent.click(screen.getByRole("switch", { name: "Resume interrupted sessions", checked: false }))

    expect(state.set).toHaveBeenCalledWith(true)
  })

  it("is hidden until pi-host answered", () => {
    state.value = null
    open()

    expect(screen.queryByRole("switch", { name: "Resume interrupted sessions" })).toBeNull()
  })
})
