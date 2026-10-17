import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SettingsMenu } from "@/components/SettingsMenu"
import { DEFAULT_TAB_CONFIG, type TabConfig } from "@/contexts/SettingsContext"

const config: TabConfig = { ...DEFAULT_TAB_CONFIG, reasoningEffort: "none" }

describe("SettingsMenu", () => {
  it("keeps domain and answer guidance inside Deep Research", () => {
    render(<SettingsMenu prompts={{}} config={config} onConfigChange={vi.fn()} models={null} researchEnabled />)

    fireEvent.click(screen.getByRole("button", { name: "Settings" }))

    expect(screen.queryByPlaceholderText("site:nature.com -reddit.com")).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText("Optional guidance for the answer…")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Deep Research" }))

    expect(screen.getByPlaceholderText("site:nature.com -reddit.com")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Optional guidance for the answer…")).toBeInTheDocument()
  })
})
