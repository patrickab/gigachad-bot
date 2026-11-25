import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { PiTranscript } from "../PiTranscript"
import { EMPTY_TRANSCRIPT } from "../transcript"
import type { CustomMessage } from "../types"

const fileLinks = { base: "/repo", open: () => {} }
const show = (message: CustomMessage) => render(<PiTranscript transcript={{ ...EMPTY_TRANSCRIPT, messages: [{ role: "user", content: "hi", timestamp: 0 }, message], entryIds: ["a", "b"] }} fileLinks={fileLinks} />)

describe("subagent message cards", () => {
  it("folds an intercom message to its sender and first line", () => {
    show({
      role: "custom", customType: "intercom_message", display: true, timestamp: 1,
      content: "**From subagent-control** (/x)\n\nlong raw text",
      details: { from: { name: "subagent-control" }, message: { content: { text: "Subagent needs attention: worker\nRun: 1" } } },
    })
    expect(screen.getByText("subagent-control")).toBeInTheDocument()
    expect(screen.getByText("Subagent needs attention: worker")).toBeInTheDocument()
    expect(screen.queryByText(/long raw text/)).toBeNull()
  })
})
