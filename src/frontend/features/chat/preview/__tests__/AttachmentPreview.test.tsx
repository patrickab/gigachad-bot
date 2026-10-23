import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import type { Attachment } from "@/lib/domain/types"
import type { NeovimSetting } from "@/contexts/SettingsContext"
import { FilePanelProvider, FilePanelSlot } from "@/features/editor/FilePanel"
import { AttachmentPreview } from "../AttachmentPreview"

const neovim = vi.hoisted(() => ({ setting: { available: false, enabled: false, setPreferred: () => {} } as NeovimSetting }))
vi.mock("@/contexts/SettingsContext", () => ({ useNeovimSetting: () => neovim.setting }))
vi.mock("@/features/editor/NeovimEditor", () => ({
  useNeovimTerminal: () => ({ terminal: null, status: { state: "running" }, restart: () => {} }),
  NeovimEditor: () => <div>neovim terminal</div>,
}))

/** The preview inside a tab's file panel. */
function renderPreview(attachment: Attachment) {
  return render(<FilePanelProvider><AttachmentPreview attachment={attachment} onClose={() => {}} /><FilePanelSlot /></FilePanelProvider>)
}

const script: Attachment = { name: "plot.py", mime: "text/x-python", url: "", content: "print('hi')\n", active: true }

describe("AttachmentPreview views", () => {
  afterEach(cleanup)

  it("offers Neovim when the backend runs it, even while this device does not prefer it", () => {
    neovim.setting = { available: true, enabled: false, setPreferred: () => {} }
    renderPreview(script)

    expect(screen.getByRole("button", { name: "Rendered" }).getAttribute("aria-pressed")).toBe("true")
    fireEvent.click(screen.getByRole("button", { name: "Neovim" }))
    expect(screen.getByText("neovim terminal")).toBeTruthy()
  })

  it("opens Neovim first when this device prefers it", () => {
    neovim.setting = { available: true, enabled: true, setPreferred: () => {} }
    renderPreview(script)

    expect(screen.getByText("neovim terminal")).toBeTruthy()
  })

  it("offers no Neovim when the backend cannot run it", () => {
    neovim.setting = { available: false, enabled: false, setPreferred: () => {} }
    renderPreview(script)

    expect(screen.queryByRole("button", { name: "Neovim" })).toBeNull()
  })
})
