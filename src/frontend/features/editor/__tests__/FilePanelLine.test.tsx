import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"

const neovim = vi.hoisted(() => ({ send: vi.fn(() => true), terminal: { attach: () => {}, detach: () => {}, fit: () => {}, restyle: () => {}, focus: () => {} } }))

vi.mock("@/contexts/SettingsContext", () => ({ useNeovimSetting: () => ({ available: true, enabled: true, setPreferred: () => {} }) }))
vi.mock("@/hooks/useRootFileDocument", () => ({
  TEXT_CODEC: {},
  useRootFileDocument: () => ({ value: "text", loadError: null, reload: () => {}, saveFromNeovim: vi.fn() }),
}))
vi.mock("../NeovimEditor", () => ({
  NeovimEditor: () => <div>neovim</div>,
  useNeovimTerminal: () => ({ terminal: neovim.terminal, status: { state: "running" }, restart: () => {}, send: neovim.send }),
}))
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

import { FilePanelProvider, FilePanelSlot, useOpenFile } from "../FilePanel"

function Opener() {
  const openFile = useOpenFile()
  return (
    <>
      <button onClick={() => openFile("/repo/a.ts", 42)}>at 42</button>
      <button onClick={() => openFile("/repo/a.ts")}>no line</button>
      <FilePanelSlot />
    </>
  )
}

afterEach(cleanup)

it("opens a file at a line in Neovim, again on each open, and without a line moves nothing", () => {
  render(<FilePanelProvider><Opener /></FilePanelProvider>)
  fireEvent.click(screen.getByRole("button", { name: "at 42" }))
  expect(neovim.send).toHaveBeenCalledWith({ type: "open_file", path: "/repo/a.ts", line: 42 })
  fireEvent.click(screen.getByRole("button", { name: "at 42" }))
  expect(neovim.send).toHaveBeenCalledTimes(2)
  fireEvent.click(screen.getByRole("button", { name: "no line" }))
  expect(neovim.send).toHaveBeenCalledTimes(2)
})
