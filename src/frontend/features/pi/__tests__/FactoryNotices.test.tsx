import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import "../FactoryNotices"
import { renderCustomMessage } from "../renderers"
import type { CustomMessage } from "../types"

const show = (content: string) =>
  render(<>{renderCustomMessage({ role: "custom", customType: "factory-notice", content, display: true, timestamp: 0 } as CustomMessage, () => <p>raw text</p>)}</>)

describe("factory notices", () => {
  it("shows the first line, the rest only when opened", () => {
    show("factory: wave board/w4 was rejected\nwave 2a9081c8  rejected\n  major  a.ts:1  broken")
    const row = screen.getByRole("button", { name: "wave board/w4 was rejected" })
    expect(screen.queryByText(/major/)).toBeNull()
    fireEvent.click(row)
    expect(screen.getByText(/major a\.ts:1 broken/)).toBeTruthy()
  })

  it("keeps a one-line notice as one line and an empty one as its text", () => {
    show("factory: wave x was merged")
    expect(screen.getByText("wave x was merged")).toBeTruthy()
    expect(screen.queryByRole("button")).toBeNull()
    show("")
    expect(screen.getByText("raw text")).toBeTruthy()
  })
})
