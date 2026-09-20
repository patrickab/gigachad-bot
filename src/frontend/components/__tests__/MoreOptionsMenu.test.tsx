import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MoreOptionsMenu } from "@/components/MoreOptionsMenu"
import { ModeProvider } from "@/hooks/useModeState"
import type { TabConfig } from "@/components/TabManager"

const config: TabConfig = {
  selectedModel: "",
  selectedPrompt: null,
  temperature: 0.2,
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

describe("MoreOptionsMenu", () => {
  it("keeps domain and answer guidance inside Deep Research", () => {
    const { container } = render(
      <ModeProvider>
        <MoreOptionsMenu prompts={{}} config={config} onConfigChange={vi.fn()} models={null} />
      </ModeProvider>,
    )

    fireEvent.click(container.querySelector("button")!)

    expect(screen.queryByPlaceholderText("site:nature.com -reddit.com")).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText("Optional guidance for the answer…")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Deep Research" }))

    expect(screen.getByPlaceholderText("site:nature.com -reddit.com")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Optional guidance for the answer…")).toBeInTheDocument()
  })
})
