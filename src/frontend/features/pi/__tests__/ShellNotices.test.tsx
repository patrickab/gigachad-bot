import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { renderCustomMessage } from "../renderers"
import { isSilentShellNotice } from "../ShellNotices"
import type { CustomMessage } from "../types"

const notice = (details: unknown): CustomMessage => ({ role: "custom", customType: "interactive-shell-transfer", content: "Session w completed. Full output: /x.log", display: true, details } as CustomMessage)
const show = (details: unknown) => render(<>{renderCustomMessage(notice(details), () => <p>raw text</p>)}</>)

describe("interactive-shell notices", () => {
  it("leaves a session that ended well out of the transcript, a failed or unreadable one in", () => {
    expect(isSilentShellNotice(notice({ sessionId: "wait-w1", duration: "4m 24s", exitCode: 0 }))).toBe(true)
    expect(isSilentShellNotice(notice({ sessionId: "build", exitCode: 2 }))).toBe(false)
    expect(isSilentShellNotice(notice(undefined))).toBe(false)
  })

  it("names a failed session's exit code", () => {
    show({ sessionId: "build", duration: "2s", exitCode: 2 })
    expect(screen.getByText("build failed (exit 2) · 2s")).toBeTruthy()
  })

  it("keeps the text of a notice it cannot read", () => {
    show(undefined)
    expect(screen.getByText("raw text")).toBeTruthy()
  })
})
