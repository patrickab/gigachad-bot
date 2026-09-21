import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useChatStream, type ChatStreamCompletion } from "@/hooks/useChatStream"
import type { ChatRequest } from "@/lib/types"
import type { SSEEvent, SSEStreamResult } from "@/lib/sse"

const TEST_MODEL = "provider/test-model"

const { createChatStream } = vi.hoisted(() => ({
  createChatStream: vi.fn(),
}))

vi.mock("@/lib/api", () => ({ createChatStream }))

function stream(events: SSEEvent[]): SSEStreamResult {
  return {
    abort: vi.fn(),
    async *[Symbol.asyncIterator]() {
      yield* events
    },
  }
}

// A stream that emits `events` then parks until abort(), so a turn can be cancelled
// mid-tool exactly as the real transport aborts an open fetch.
function parkedStream(events: SSEEvent[]): { result: SSEStreamResult; parked: Promise<void> } {
  let reachedPark!: () => void
  const parked = new Promise<void>((resolve) => { reachedPark = resolve })
  let failPark: ((err: Error) => void) | null = null
  const result: SSEStreamResult = {
    abort: vi.fn(() => {
      const err = new Error("aborted")
      err.name = "AbortError"
      failPark?.(err)
    }),
    async *[Symbol.asyncIterator]() {
      yield* events
      reachedPark()
      await new Promise<never>((_, reject) => { failPark = reject })
    },
  }
  return { result, parked }
}

function event(name: string, data: unknown): SSEEvent {
  return { event: name, data: typeof data === "string" ? data : JSON.stringify(data) }
}

function request(user_msg: string): ChatRequest {
  return {
    model: TEST_MODEL,
    chat_id: "chat-1",
    user_msg,
    img_paths: [],
  }
}

beforeEach(() => {
  createChatStream.mockReset()
})

describe("useChatStream", () => {
  it("reconciles a tool result by call id without losing answer text or usage", async () => {
    createChatStream.mockReturnValue(stream([
      event("token", "Before "),
      event("tool_call", { id: "call-1", name: "web_search", arguments: { query: "cats" } }),
      event("tool_progress", {
        id: "call-1",
        stages: [{ id: "stage-1", label: "Planning search", status: "done", started_at: 1, duration: 0.2 }],
      }),
      event("token", "after"),
      event("tool_result", {
        id: "call-1",
        name: "web_search",
        summary: "1 source",
        detail: { query: "cats" },
        error: null,
      }),
      event("usage", { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 }),
      event("done", ""),
    ]))
    const { result } = renderHook(() => useChatStream())
    let completion: ChatStreamCompletion | null = null

    await act(async () => {
      completion = await result.current.send(request("Find cats"))
    })

    expect(result.current.messages).toEqual([
      { role: "user", content: "Find cats" },
      {
        role: "assistant",
        content: "Before after",
        tool_calls: [{
          id: "call-1",
          name: "web_search",
          arguments: { query: "cats" },
          status: "done",
          summary: "1 source",
          detail: { query: "cats" },
          error: null,
          stages: [{ id: "stage-1", label: "Planning search", status: "done", started_at: 1, duration: 0.2 }],
        }],
      },
    ])
    expect(result.current.totalUsage).toEqual({
      prompt_tokens: 11,
      completion_tokens: 7,
      total_tokens: 18,
    })
    expect(completion).toEqual({
      messages: result.current.messages,
      usage: result.current.totalUsage,
    })
  })

  it("sends only user and assistant text as history on a later turn", async () => {
    const detailSentinel = "DETAIL_SENTINEL"
    const sourceSentinel = "SOURCE_SENTINEL"
    const sandboxSentinel = "SANDBOX_SENTINEL"
    createChatStream
      .mockReturnValueOnce(stream([
        event("tool_call", { id: "call-1", name: "web_search", arguments: { query: "weather" } }),
        event("tool_result", {
          id: "call-1",
          name: "web_search",
          summary: "Browser-only result",
          detail: { raw: detailSentinel },
          sources: [{ title: sourceSentinel, url: "https://example.test", content: "" }],
          sandbox: {
            status: "completed",
            manifest_id: "manifest-1",
            workspace_changed: false,
            outputs: [{
              display_id: "output-1",
              title: null,
              mime_bundle: {},
              text: sandboxSentinel,
            }],
          },
          error: null,
        }),
        event("token", "Visible answer"),
        event("done", ""),
      ]))
      .mockReturnValueOnce(stream([
        event("token", "Second answer"),
        event("done", ""),
      ]))
    const { result } = renderHook(() => useChatStream())

    await act(async () => {
      await result.current.send(request("Initial question"))
    })

    expect(JSON.stringify(result.current.messages[1].tool_calls)).toContain(detailSentinel)
    expect(JSON.stringify(result.current.messages[1].tool_calls)).toContain(sourceSentinel)
    expect(JSON.stringify(result.current.messages[1].tool_calls)).toContain(sandboxSentinel)

    await act(async () => {
      await result.current.send(request("Follow up"))
    })

    const laterRequest = createChatStream.mock.calls[1][0]
    expect(laterRequest.messages).toEqual([
      { role: "user", content: "Initial question" },
      { role: "assistant", content: "Visible answer" },
      { role: "user", content: "Follow up" },
    ])
    expect(JSON.stringify(laterRequest.messages)).not.toContain(detailSentinel)
    expect(JSON.stringify(laterRequest.messages)).not.toContain(sourceSentinel)
    expect(JSON.stringify(laterRequest.messages)).not.toContain(sandboxSentinel)
  })

  it("leaves no running tool record when the turn is aborted", async () => {
    const { result: parkedResult, parked } = parkedStream([
      event("tool_call", { id: "call-1", name: "sandbox_plot", arguments: { code: "plot()" } }),
      event("tool_progress", {
        id: "call-1",
        stages: [{ id: "stage-1", label: "Generating chart", status: "running", started_at: Date.now() / 1_000, duration: 0 }],
      }),
    ])
    createChatStream.mockReturnValue(parkedResult)
    const { result } = renderHook(() => useChatStream())
    let completion: ChatStreamCompletion | null = null

    await act(async () => {
      const sent = result.current.send(request("Plot it"))
      await parked
      result.current.cancel()
      completion = await sent
    })

    expect(completion).toBeNull()
    const calls = result.current.messages[1].tool_calls
    expect(calls).toHaveLength(1)
    expect(calls?.[0]).toMatchObject({ id: "call-1", status: "error", error: "Cancelled" })
    expect(calls?.[0].stages?.[0]).toMatchObject({ status: "error" })
    expect(calls?.every(call => call.status !== "running")).toBe(true)
  })

  it("terminates a pending tool record on an error event", async () => {
    createChatStream.mockReturnValue(stream([
      event("tool_call", { id: "call-1", name: "web_search", arguments: { query: "cats" } }),
      event("error", "upstream died"),
    ]))
    const { result } = renderHook(() => useChatStream())

    await act(async () => {
      await result.current.send(request("Find cats"))
    })

    const assistant = result.current.messages[1]
    expect(assistant.content).toContain("upstream died")
    expect(assistant.tool_calls?.[0]).toMatchObject({ id: "call-1", status: "error", error: "upstream died" })
  })
})
