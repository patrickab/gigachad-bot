import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SettingsProvider, useHostAccess, useSettings } from "../SettingsContext"

vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchBackendConfig: vi.fn(async () => ({
    default_model: "backend-default",
    vision_model: "vision",
    neovim_available: false,
    learnharness_url: null,
    host_access: true,
    animated_background: false,
  })),
  fetchPiSettings: vi.fn(async () => ({ autoResume: false })),
}))

function Probe() {
  const hostAccess = useHostAccess()
  const { tabDefaults } = useSettings()
  return <output>{`${tabDefaults.selectedModel || "none"} ${hostAccess ? "host" : "guest"}`}</output>
}

describe("SettingsProvider", () => {
  it("derives tab defaults and host access from the backend config", async () => {
    render(<SettingsProvider><Probe /></SettingsProvider>)
    expect(screen.getByRole("status")).toHaveTextContent("none guest")
    expect(await screen.findByText("backend-default host")).toBeInTheDocument()
  })
})
