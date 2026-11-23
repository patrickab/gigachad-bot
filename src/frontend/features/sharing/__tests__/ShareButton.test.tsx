import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const api = vi.hoisted(() => ({
  readShareAccess: vi.fn(),
  createShare: vi.fn(),
  revokeShare: vi.fn(),
  shareLinkUrl: (secret: string) => `https://app.test/share#${secret}`,
}))
vi.mock("@/lib/backend/api", () => api)

import { ShareButton } from "@/features/sharing/ShareButton"

const TARGET = { kind: "document" as const, key: "project/p/document/board.canvas" }
const writeText = vi.fn(() => Promise.resolve())

function open() {
  render(<ShareButton target={TARGET} />)
  fireEvent.click(screen.getByRole("button", { name: "Share" }))
}

describe("ShareButton", () => {
  beforeEach(() => {
    vi.resetAllMocks()
    Object.assign(navigator, { clipboard: { writeText } })
    api.revokeShare.mockResolvedValue(undefined)
  })

  it("creates a link, copies and shows its URL, then revokes it", async () => {
    api.readShareAccess.mockResolvedValue(null)
    api.createShare.mockResolvedValue({ secret: "s3cret", access: "edit" })
    open()

    fireEvent.click(await screen.findByRole("button", { name: "Create link" }))

    expect(await screen.findByRole("textbox", { name: "Share link" })).toHaveValue("https://app.test/share#s3cret")
    expect(screen.getByText("Anyone with the link can edit")).toBeInTheDocument()
    expect(api.createShare).toHaveBeenCalledWith(TARGET)
    expect(writeText).toHaveBeenCalledWith("https://app.test/share#s3cret")

    fireEvent.click(screen.getByRole("button", { name: "Revoke" }))
    expect(await screen.findByRole("button", { name: "Create link" })).toBeInTheDocument()
    expect(api.revokeShare).toHaveBeenCalledWith(TARGET)
  })

  it("offers to replace a live link whose secret this page never saw", async () => {
    api.readShareAccess.mockResolvedValue("view")
    open()

    expect(await screen.findByText("Anyone with the link can view")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "New link" })).toBeInTheDocument()
    expect(screen.queryByRole("textbox", { name: "Share link" })).not.toBeInTheDocument()
  })
})
