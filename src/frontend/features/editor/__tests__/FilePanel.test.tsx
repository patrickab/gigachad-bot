import { useState } from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { FilePanel, FilePanelProvider, FilePanelSlot, FilePanelToggle, useOpenFile } from "../FilePanel"

vi.mock("@/contexts/SettingsContext", () => ({ useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }) }))
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

/** An attachment-like panel owner next to a file link, both in one tab. */
function Tab() {
  const [attachmentOpen, setAttachmentOpen] = useState(true)
  const openFile = useOpenFile()
  return (
    <>
      {attachmentOpen && <FilePanel label="notes.md" onClose={() => setAttachmentOpen(false)}>attachment body</FilePanel>}
      <button onClick={() => openFile("/repo/plot.png")}>open file</button>
      <FilePanelSlot />
    </>
  )
}

afterEach(cleanup)

it("shows one item at a time: opening a file closes the shown attachment", () => {
  render(<FilePanelProvider><Tab /></FilePanelProvider>)
  expect(screen.getByText("attachment body")).toBeVisible()

  fireEvent.click(screen.getByRole("button", { name: "open file" }))
  expect(screen.queryByText("attachment body")).toBeNull()
  expect(screen.getByRole("img", { name: "plot.png" })).toBeInTheDocument()

  fireEvent.click(screen.getByRole("button", { name: "Close panel" }))
  expect(screen.queryByRole("region", { name: "/repo/plot.png" })).toBeNull()
})

it("collapsing keeps the content mounted, so a running Neovim survives", () => {
  render(<FilePanelProvider><Tab /><FilePanelToggle /></FilePanelProvider>)
  fireEvent.click(screen.getByRole("button", { name: "Close file panel" }))
  expect(screen.getByText("attachment body")).toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "Open file panel" }))
  expect(screen.getByText("attachment body")).toBeVisible()
})

it("opening the shown file again expands a collapsed panel", () => {
  render(<FilePanelProvider><Tab /><FilePanelToggle /></FilePanelProvider>)
  fireEvent.click(screen.getByRole("button", { name: "open file" }))
  fireEvent.click(screen.getByRole("button", { name: "Close file panel" }))
  fireEvent.click(screen.getByRole("button", { name: "open file" }))
  expect(screen.getByRole("button", { name: "Close file panel" })).toBeInTheDocument()
})
