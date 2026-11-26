import { afterEach, describe, expect, it } from "vitest"
import { cleanup, render, screen } from "@testing-library/react"
import { HeldMessages } from "../HeldMessages"
import type { HeldMessage } from "../outbox"

afterEach(cleanup)

describe("HeldMessages", () => {
  it("shows the images of a held message", () => {
    const entry: HeldMessage = {
      id: "held-1", text: "look", mode: "steer", state: "sending", dispatched: false, createdAt: 0,
      images: [{ type: "image", data: "aGk=", mimeType: "image/png" }],
    }
    render(<HeldMessages held={[entry]} onRetry={() => {}} onDelete={() => {}} />)
    expect(screen.getByText("look")).toBeTruthy()
    expect(document.querySelector('img[src^="data:image/png;base64,aGk="]')).not.toBeNull()
  })
})
