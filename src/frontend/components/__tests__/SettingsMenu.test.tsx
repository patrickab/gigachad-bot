import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SettingsMenu } from "@/components/SettingsMenu"
import { ModeProvider } from "@/hooks/useModeState"
import type { TabConfig } from "@/components/TabManager"

const config: TabConfig = {
  selectedModel: "",
  selectedPrompt: null,
  reasoningEffort: "none",
  downscaleImages: true,
  researchFastModel: "",
  researchSmartModel: "",
  researchStrategicModel: "",
  researchDepth: 2,
  researchBreadth: 4,
  researchReasoning: "medium",
  researchReportType: "deep",
  searchSystemInstructions: "",
  searchDomain: "",
}

describe("SettingsMenu", () => {
  it("keeps domain and answer guidance inside Deep Research", () => {
    render(
      <ModeProvider>
        <SettingsMenu prompts={{}} config={config} onConfigChange={vi.fn()} models={null} />
      </ModeProvider>,
    )

    fireEvent.click(screen.getByRole("button", { name: "Settings" }))

    expect(screen.queryByPlaceholderText("site:nature.com -reddit.com")).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText("Optional guidance for the answer…")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Deep Research" }))

    expect(screen.getByPlaceholderText("site:nature.com -reddit.com")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Optional guidance for the answer…")).toBeInTheDocument()
  })
})
