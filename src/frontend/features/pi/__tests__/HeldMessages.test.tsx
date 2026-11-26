import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { HeldMessages } from "../HeldMessages"
import type { HeldMessage } from "../outbox"

afterEach(cleanup)

describe("HeldMessages", () => {
  it("shows the images of a held message", () => {
    const entry: HeldMessage = {
      id: "held-1", text: "look", mode: "steer", state: "sending", dispatched: false, createdAt: 0,
      images: [{ type: "image", data: "aGk=", mimeType: "image/png" }],
    }
    const onOpenImage = vi.fn()
    render(<HeldMessages held={[entry]} onRetry={() => {}} onDelete={() => {}} onOpenImage={onOpenImage} />)
    expect(screen.getByText("look")).toBeTruthy()
    const image = document.querySelector('img[src^="data:image/png;base64,aGk="]')
    expect(image).not.toBeNull()
    fireEvent.click(image!.closest("button")!)
    expect(onOpenImage).toHaveBeenCalledWith(expect.objectContaining({ url: "data:image/png;base64,aGk=" }))
  })
})
