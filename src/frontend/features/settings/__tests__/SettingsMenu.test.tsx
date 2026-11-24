import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SettingsMenu } from "@/features/settings/SettingsMenu"
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

  it("switches glass mode on <html> and remembers it", async () => {
    const stored = new Map<string, string>()
    vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => void stored.set(key, value) })
    render(<SettingsMenu prompts={{}} config={config} onConfigChange={vi.fn()} models={null} researchEnabled={false} />)
    fireEvent.click(screen.getByRole("button", { name: "Settings" }))
    const glass = screen.getByRole("switch", { name: "Glass" })
    expect(glass).toHaveAttribute("aria-checked", "false")

    fireEvent.click(glass)

    expect(document.documentElement.classList.contains("glass")).toBe(true)
    expect(localStorage.getItem("glass")).toBe("on")
    expect(await screen.findByRole("switch", { name: "Glass", checked: true })).toBeInTheDocument()

    fireEvent.click(glass)
    expect(document.documentElement.classList.contains("glass")).toBe(false)
    expect(localStorage.getItem("glass")).toBe("off")
    vi.unstubAllGlobals()
  })
})
