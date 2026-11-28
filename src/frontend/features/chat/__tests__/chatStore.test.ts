import { beforeEach, describe, expect, it, vi } from "vitest"
import { createChatStore, type ChatStore } from "@/features/chat/chatStore"
import type { ChatRequest, Message } from "@/lib/domain/types"
import type { SSEEvent, SSEStreamResult } from "@/lib/backend/sse"

const { createChatStream } = vi.hoisted(() => ({ createChatStream: vi.fn() }))

vi.mock("@/lib/backend/api", () => ({ createChatStream, currentConnectionPool: () => 0, leaveConnectionPool: vi.fn() }))
// Every token lands at once
vi.mock("@/lib/backend/tokenPacer", () => ({
  createTokenPacer: (emit: (text: string) => void) => ({ push: emit, flush: () => {}, finish: async () => {} }),
}))

function request(user_msg: string): ChatRequest {
  return { model: "provider/test-model", chat_id: "chat-1", user_msg, img_paths: [] }
}

/** Append the question and its empty answer, then stream the reply into it, as `handleSend` does. */
function ask(chat: ChatStore, user_msg: string) {
  const answer: Message = { role: "assistant", content: "" }
  chat.update((current) => [...current, { role: "user", content: user_msg }, answer])
  return chat.send(request(user_msg), answer)
}

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

/** A stream that yields `events`, then waits for `release()`, then ends with `done`; `abort` fails the wait. */
function heldStream(events: SSEEvent[]) {
  let release!: () => void
  let fail!: (error: Error) => void
  let reached!: () => void
  const held = new Promise<void>((resolve) => { reached = resolve })
  const stream: SSEStreamResult = {
    abort: vi.fn(() => fail(Object.assign(new Error("aborted"), { name: "AbortError" }))),
    async *[Symbol.asyncIterator]() {
      yield* events
      reached()
      await new Promise<void>((resolve, reject) => { release = resolve; fail = reject })
      yield { event: "token", data: " late" }
      yield { event: "done", data: "" }
    },
  }
  return { stream, held, release: () => release() }
}

beforeEach(() => createChatStream.mockReset())

describe("createChatStore", () => {
  it("sends what a write right before send left in the chat", async () => {
    const { stream, held, release } = heldStream([])
    createChatStream.mockReturnValue(stream)
    const chat = createChatStore()
    chat.update(() => [{ role: "user", content: "Earlier" }, { role: "assistant", content: "Answer" }])

    const sent = ask(chat, "Question")

    expect(createChatStream.mock.calls[0][0].messages).toEqual([
      { role: "user", content: "Earlier" },
      { role: "assistant", content: "Answer" },
    ])
    await held
    release()
    await sent
  })

  it("writes nothing into a chat emptied while the reply streams", async () => {
    const { stream, held, release } = heldStream([{ event: "token", data: "Early" }])
    createChatStream.mockReturnValue(stream)
    const chat = createChatStore()
    const sent = ask(chat, "Question")
    await held
    expect(chat.get().messages.at(-1)).toEqual({ role: "assistant", content: "Early" })

    chat.update(() => [])
    release()

    expect(await sent).toBeNull()
    expect(chat.get().messages).toEqual([])
  })

  it("a send whose placeholder is no longer the last message writes nothing and resolves null", async () => {
    createChatStream.mockReturnValue(stream([event("token", "Orphan"), event("done", "")]))
    const chat = createChatStore()

    expect(await chat.send(request("Question"), { role: "assistant", content: "" })).toBeNull()
    expect(chat.get().messages).toEqual([])
  })

  it("reset clears messages and usage in one notification", async () => {
    createChatStream.mockReturnValue(stream([event("token", "Hi"), event("usage", JSON.stringify({ prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 })), event("done", "")]))
    const chat = createChatStore()
    await ask(chat, "Question")
    const seen: unknown[] = []
    chat.subscribe(() => seen.push({ messages: chat.get().messages, usage: chat.get().usage }))

    chat.reset()

    expect(seen).toEqual([{ messages: [], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } }])
  })

  it("stops streaming at once on cancel", async () => {
    const { stream, held } = heldStream([])
    createChatStream.mockReturnValue(stream)
    const chat = createChatStore()
    const sent = ask(chat, "Question")
    await held
    expect(chat.get().isStreaming).toBe(true)

    chat.cancel()

    expect(chat.get().isStreaming).toBe(false)
    expect(stream.abort).toHaveBeenCalled()
    expect(await sent).toBeNull()
  })
})

describe("a chat store's stream", () => {
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
    const chat = createChatStore()
    const completion = await ask(chat, "Find cats")

    expect(chat.get().messages).toEqual([
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
    expect(chat.get().usage).toEqual({
      prompt_tokens: 11,
      completion_tokens: 7,
      total_tokens: 18,
    })
    expect(completion).toEqual({
      messages: chat.get().messages,
      usage: chat.get().usage,
    })
  })

  it("publishes the first token before the response finishes", async () => {
    let markFirstToken!: () => void
    let finish!: () => void
    const firstToken = new Promise<void>((resolve) => { markFirstToken = resolve })
    const responseFinished = new Promise<void>((resolve) => { finish = resolve })
    createChatStream.mockReturnValue({
      abort: vi.fn(),
      async *[Symbol.asyncIterator]() {
        yield event("token", "First")
        markFirstToken()
        await responseFinished
        yield event("token", " second")
        yield event("done", "")
      },
    })
    const chat = createChatStore()
    const sent = ask(chat, "Stream this")
    await firstToken
    await vi.waitFor(() => expect(chat.get().messages.at(-1)).toEqual({ role: "assistant", content: "First" }))

    finish()
    await sent
  })

  it("keeps usage for the current request instead of accumulating prior turns", async () => {
    createChatStream
      .mockReturnValueOnce(stream([
        event("token", "First answer"),
        event("usage", { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 }),
        event("done", ""),
      ]))
      .mockReturnValueOnce(stream([
        event("token", "Second answer"),
        event("usage", { prompt_tokens: 29, completion_tokens: 13, total_tokens: 42 }),
        event("done", ""),
      ]))
    const chat = createChatStore()

    await ask(chat, "First question")
    await ask(chat, "Second question")

    expect(chat.get().usage).toEqual({
      prompt_tokens: 29,
      completion_tokens: 13,
      total_tokens: 42,
    })
  })

  it("sends the rendered mindmap as assistant context without duplicating it in chat", async () => {
    const mindmap = "```markmap\n# Topic\n```"
    createChatStream
      .mockReturnValueOnce(stream([
        event("tool_call", { id: "call-mindmap", name: "mindmap", arguments: {} }),
        event("tool_result", {
          id: "call-mindmap",
          name: "mindmap",
          summary: "Mind map ready",
          detail: { mindmap },
          error: null,
        }),
        event("done", ""),
      ]))
      .mockReturnValueOnce(stream([
        event("token", "Second answer"),
        event("done", ""),
      ]))
    const chat = createChatStore()

    await ask(chat, "Create a mind map")
    expect(chat.get().messages[1].content).toBe("")

    await ask(chat, "Follow up")

    expect(createChatStream.mock.calls[1][0].messages).toEqual([
      { role: "user", content: "Create a mind map" },
      { role: "assistant", content: mindmap },
    ])
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
    const chat = createChatStore()

    await ask(chat, "Initial question")

    expect(JSON.stringify(chat.get().messages[1].tool_calls)).toContain(detailSentinel)
    expect(JSON.stringify(chat.get().messages[1].tool_calls)).toContain(sourceSentinel)
    expect(JSON.stringify(chat.get().messages[1].tool_calls)).toContain(sandboxSentinel)

    await ask(chat, "Follow up")

    const laterRequest = createChatStream.mock.calls[1][0]
    expect(laterRequest.messages).toEqual([
      { role: "user", content: "Initial question" },
      { role: "assistant", content: "Visible answer" },
    ])
    expect(JSON.stringify(laterRequest.messages)).not.toContain(detailSentinel)
    expect(JSON.stringify(laterRequest.messages)).not.toContain(sourceSentinel)
    expect(JSON.stringify(laterRequest.messages)).not.toContain(sandboxSentinel)
  })

  it("sends the request again when the stream died before any event", async () => {
    const dead: SSEStreamResult = {
      abort: vi.fn(),
      async *[Symbol.asyncIterator]() { throw new TypeError("NetworkError when attempting to fetch resource.") },
    }
    createChatStream.mockReturnValueOnce(dead).mockReturnValueOnce(stream([event("token", "Hi"), event("done", "")]))
    const chat = createChatStore()
    const completion = await ask(chat, "hi")
    expect(createChatStream).toHaveBeenCalledTimes(2)
    expect(completion).not.toBeNull()
    expect(chat.get().messages[1].content).toBe("Hi")
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
    const chat = createChatStore()
    const sent = ask(chat, "Plot it")
    await parked
    chat.cancel()
    const completion = await sent

    expect(completion).toBeNull()
    const calls = chat.get().messages[1].tool_calls
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
    const chat = createChatStore()

    await ask(chat, "Find cats")

    const assistant = chat.get().messages[1]
    expect(assistant.content).toContain("upstream died")
    expect(assistant.tool_calls?.[0]).toMatchObject({ id: "call-1", status: "error", error: "upstream died" })
  })

})
