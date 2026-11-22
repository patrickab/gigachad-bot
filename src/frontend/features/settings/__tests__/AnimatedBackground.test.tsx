import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SettingsMenu } from "@/features/settings/SettingsMenu"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"

const state = vi.hoisted(() => ({ value: true as boolean | null, set: vi.fn() }))

vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/contexts/SettingsContext")>()),
  useAnimatedBackground: () => state.value,
  useSetAnimatedBackground: () => state.set,
}))

describe("animated background setting", () => {
  it("switches the stored setting", () => {
    render(<SettingsMenu prompts={{}} config={DEFAULT_TAB_CONFIG} onConfigChange={vi.fn()} models={null} researchEnabled={false} />)
    fireEvent.click(screen.getByRole("button", { name: "Settings" }))

    fireEvent.click(screen.getByRole("switch", { name: "Animated background", checked: true }))

    expect(state.set).toHaveBeenCalledWith(false)
  })
})
