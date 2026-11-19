import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { PiTranscript } from "../PiTranscript"
import { registerMessageRenderer, registerToolRenderer, renderCustomMessage, renderToolCall, toolPlacement } from "../renderers"
import { EMPTY_TRANSCRIPT, type Transcript } from "../transcript"
import type { AssistantMessage, CustomMessage, ToolCall, UserMessage } from "../types"

const fileLinks = { base: "/repo", open: () => {} }
const todoCall: ToolCall = { type: "toolCall", id: "call-1", name: "todo", arguments: {} }
const unregister: (() => void)[] = []

afterEach(() => {
  for (const remove of unregister.splice(0)) remove()
})

describe("tool renderer registry", () => {
  it("resolves the newest renderer first, next() reaching the older ones and then the default", () => {
    unregister.push(registerToolRenderer("todo", (_call, _run, next) => <span>old({next()})</span>))
    unregister.push(registerToolRenderer("todo", (_call, _run, next) => <span>new({next()})</span>))
    const { container } = render(<>{renderToolCall(todoCall, undefined, () => "default")}</>)
    expect(container.textContent).toBe("new(old(default))")
  })

  it("passes per call and falls back to the default for unregistered tools", () => {
    unregister.push(registerToolRenderer("todo", (call, _run, next) => (call.arguments.mine ? "mine" : next())))
    expect(renderToolCall({ ...todoCall, arguments: { mine: true } }, undefined, () => "default")).toBe("mine")
    expect(renderToolCall(todoCall, undefined, () => "default")).toBe("default")
    expect(renderToolCall({ ...todoCall, name: "bash" }, undefined, () => "default")).toBe("default")
  })

  it("drops a renderer on unregister, the newest registration decides the placement", () => {
    const remove = registerToolRenderer("todo", () => "custom", { placement: "standalone" })
    expect(toolPlacement("todo")).toBe("standalone")
    remove()
    expect(toolPlacement("todo")).toBe("inline")
    expect(renderToolCall(todoCall, undefined, () => "default")).toBe("default")
  })
})

describe("message renderer registry", () => {
  const card: CustomMessage = { role: "custom", customType: "video-preview", content: "frames", display: true, timestamp: 1 }

  it("renders a customType through its chain, falling back to the text", () => {
    unregister.push(registerMessageRenderer("video-preview", (message, next) => (message.content === "frames" ? "card" : next())))
    expect(renderCustomMessage(card, () => "text")).toBe("card")
    expect(renderCustomMessage({ ...card, content: "other" }, () => "text")).toBe("text")
    expect(renderCustomMessage({ ...card, customType: "other" }, () => "text")).toBe("text")
  })
})

describe("PiTranscript with renderers", () => {
  const prompt: UserMessage = { role: "user", content: "Plan it", timestamp: 1 }
  const reply: AssistantMessage = {
    role: "assistant",
    content: [{ type: "toolCall", id: "call-1", name: "todo", arguments: {} }, { type: "toolCall", id: "call-2", name: "bash", arguments: { command: "ls" } }, { type: "text", text: "Planned" }],
    provider: "anthropic",
    model: "claude",
    stopReason: "stop",
    timestamp: 2,
  }
  const notice: CustomMessage = { role: "custom", customType: "video-preview", content: "frames", display: true, timestamp: 3 }
  const settled: Transcript = { ...EMPTY_TRANSCRIPT, messages: [prompt, reply, notice], entryIds: ["a", "b", "c"] }

  it("keeps standalone tools outside the folded Process details, inline ones inside", () => {
    unregister.push(registerToolRenderer("todo", () => <div>Todo card</div>, { placement: "standalone" }))
    unregister.push(registerToolRenderer("bash", (_call, _run, next) => <div>Wrapped {next()}</div>))
    render(<PiTranscript transcript={settled} fileLinks={fileLinks} />)
    expect(screen.getByText("Todo card")).toBeTruthy()
    expect(screen.queryByText("Wrapped", { exact: false })).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: /Process details/ }))
    expect(screen.getByText("Wrapped", { exact: false })).toBeTruthy()
  })

  it("renders a registered customType as its card", () => {
    unregister.push(registerMessageRenderer("video-preview", () => <div>Video card</div>))
    render(<PiTranscript transcript={settled} fileLinks={fileLinks} />)
    expect(screen.getByText("Video card")).toBeTruthy()
  })
})
