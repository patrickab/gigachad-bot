/** Gateway routes over pi-ai's faux provider: no real model, no Pi runtime, no network. */
import { beforeEach, describe, expect, test } from "bun:test";
import {
  type AssistantMessage,
  createModels,
  type FauxProviderHandle,
  fauxAssistantMessage,
  fauxProvider,
  fauxText,
  fauxToolCall,
  type Models,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import { createGateway, type GatewayHandler } from "../src/gateway/index.ts";
import { mapAnthropicUsage } from "../src/gateway/usage.ts";

let faux: FauxProviderHandle;
let gateway: GatewayHandler;
let usageCalls: number;

beforeEach(() => {
  faux = fauxProvider({ provider: "anthropic", models: [{ id: "claude-test", reasoning: true }, { id: "plain" }] });
  const codex = fauxProvider({ provider: "openai-codex", models: [{ id: "gpt-test" }] });
  const models = createModels();
  models.setProvider(faux.provider);
  models.setProvider(codex.provider);
  usageCalls = 0;
  gateway = createGateway(
    async () => models,
    async () => {
      usageCalls++;
      return {};
    },
  );
});

async function call(method: string, path: string, body?: unknown): Promise<Response> {
  const url = new URL(`http://pi-host${path}`);
  const init = body === undefined ? { method } : { method, body: JSON.stringify(body), headers: { "content-type": "application/json" } };
  const response = await gateway(new Request(url, init), url);
  if (!response) throw new Error(`no route for ${method} ${path}`);
  return response;
}

type ToolDelta = { index: number; id?: string; type?: string; function: { name?: string; arguments: string } };
type Delta = { role?: string; content?: string; reasoning_content?: string; thinking_blocks?: unknown[]; tool_calls?: ToolDelta[] };
type Chunk = { choices: { index: number; delta: Delta; finish_reason: string | null }[]; usage?: { total_tokens: number } };

/** The JSON chunks of an SSE body, after checking it ends with `[DONE]`. */
async function sseChunks(response: Response): Promise<Chunk[]> {
  const frames = (await response.text()).split("\n\n").filter(Boolean).map((frame) => frame.replace(/^data: /, ""));
  expect(frames.at(-1)).toBe("[DONE]");
  return frames.slice(0, -1).map((data) => JSON.parse(data) as Chunk);
}

const user = (content: string) => ({ role: "user", content });
const signedThinking = { type: "thinking" as const, thinking: "let me think", thinkingSignature: "sig-1" };

describe("GET /v1/models", () => {
  test("lists D13 providers with their thinking ladder, no Codex", async () => {
    const body = await (await call("GET", "/v1/models")).json();
    const byId = Object.fromEntries(body.data.map((model: { id: string }) => [model.id, model]));
    expect(Object.keys(byId).sort()).toEqual(["anthropic/claude-test", "anthropic/plain"]);
    expect(byId["anthropic/claude-test"].thinking_levels).toContain("high");
    expect(byId["anthropic/plain"].thinking_levels).toEqual([]);
    expect(byId["anthropic/plain"]).toMatchObject({ object: "model", owned_by: "anthropic", reasoning: false });
  });
});

describe("POST /v1/chat/completions", () => {
  test("non-stream returns an OpenAI completion", async () => {
    faux.setResponses([fauxAssistantMessage("hello there")]);
    const response = await call("POST", "/v1/chat/completions", { model: "anthropic/claude-test", messages: [user("hi")] });
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.object).toBe("chat.completion");
    expect(body.model).toBe("anthropic/claude-test");
    expect(body.choices[0]).toMatchObject({ message: { role: "assistant", content: "hello there" }, finish_reason: "stop" });
    expect(body.usage.total_tokens).toBe(body.usage.prompt_tokens + body.usage.completion_tokens);
  });

  test("stream sends role, content deltas, finish, usage chunk, then [DONE]", async () => {
    faux.setResponses([fauxAssistantMessage("streamed answer with several tokens")]);
    const response = await call("POST", "/v1/chat/completions", {
      model: "claude-test",
      messages: [user("hi")],
      stream: true,
      stream_options: { include_usage: true },
    });
    expect(response.headers.get("content-type")).toBe("text/event-stream");
    const chunks = await sseChunks(response);
    const usage = chunks.at(-1)!;
    expect(usage.choices).toEqual([]);
    expect(usage.usage!.total_tokens).toBeGreaterThan(0);
    expect(chunks.at(-2)!.choices[0]).toEqual({ index: 0, delta: {}, finish_reason: "stop" });
    expect(chunks[0]!.choices[0]!.delta).toEqual({ role: "assistant", content: "" });
    const text = chunks.slice(1, -2).map((chunk) => chunk.choices[0]!.delta.content).join("");
    expect(text).toBe("streamed answer with several tokens");
  });

  test("stream without include_usage has no usage chunk", async () => {
    faux.setResponses([fauxAssistantMessage("x")]);
    const chunks = await sseChunks(
      await call("POST", "/v1/chat/completions", { model: "anthropic/plain", messages: [user("hi")], stream: true }),
    );
    expect(chunks.filter((chunk) => chunk.usage)).toEqual([]);
  });

  test("tool calls stream as indexed deltas and finish with tool_calls", async () => {
    faux.setResponses([
      fauxAssistantMessage([fauxText("checking"), fauxToolCall("get_time", { timezone: "Asia/Tokyo" }, { id: "call_1" })], {
        stopReason: "toolUse",
      }),
    ]);
    const tools = [
      {
        type: "function",
        function: { name: "get_time", parameters: { type: "object", properties: { timezone: { type: "string" } } } },
      },
    ];
    const chunks = await sseChunks(
      await call("POST", "/v1/chat/completions", {
        model: "anthropic/claude-test",
        messages: [user("time in Tokyo?")],
        tools,
        tool_choice: "required",
        stream: true,
      }),
    );
    const toolDeltas = chunks.flatMap((chunk) => chunk.choices[0]?.delta.tool_calls ?? []);
    expect(toolDeltas[0]).toEqual({ index: 0, id: "call_1", type: "function", function: { name: "get_time", arguments: "" } });
    expect(toolDeltas.every((delta) => delta.index === 0)).toBe(true);
    const args = toolDeltas.map((delta) => delta.function.arguments).join("");
    expect(JSON.parse(args)).toEqual({ timezone: "Asia/Tokyo" });
    expect(chunks.at(-1)!.choices[0]!.finish_reason).toBe("tool_calls");
  });

  test("signed thinking_blocks go out (stream and non-stream) and come back in", async () => {
    let seen: TranscriptContext | undefined;
    faux.setResponses([
      fauxAssistantMessage([signedThinking, fauxText("answer")]),
      fauxAssistantMessage([signedThinking, fauxText("answer")]),
      (context) => {
        seen = context;
        return fauxAssistantMessage("second turn");
      },
    ]);
    const request = { model: "anthropic/claude-test", messages: [user("think")], reasoning_effort: "high" };
    const body = await (await call("POST", "/v1/chat/completions", request)).json();
    const message = body.choices[0].message;
    expect(message.reasoning_content).toBe("let me think");
    expect(message.thinking_blocks).toEqual([{ type: "thinking", thinking: "let me think", signature: "sig-1" }]);

    const chunks = await sseChunks(await call("POST", "/v1/chat/completions", { ...request, stream: true }));
    const blockChunks = chunks.filter((chunk) => chunk.choices[0]?.delta.thinking_blocks);
    expect(blockChunks).toHaveLength(1);
    expect(blockChunks[0]!.choices[0]!.delta).toEqual({
      reasoning_content: "",
      thinking_blocks: [{ type: "thinking", thinking: "let me think", signature: "sig-1" }],
    });

    await call("POST", "/v1/chat/completions", {
      ...request,
      messages: [user("think"), { role: "assistant", content: "answer", thinking_blocks: message.thinking_blocks }, user("again")],
    });
    const assistant = seen?.messages.find((entry) => entry.role === "assistant") as AssistantMessage | undefined;
    expect(assistant?.content[0]).toMatchObject({ type: "thinking", thinking: "let me think", thinkingSignature: "sig-1" });
  });

  test("unknown or unlisted model is a 404", async () => {
    const unknown = await call("POST", "/v1/chat/completions", { model: "anthropic/nope", messages: [user("hi")] });
    expect(unknown.status).toBe(404);
    const codex = await call("POST", "/v1/chat/completions", { model: "openai-codex/gpt-test", messages: [user("hi")] });
    expect(codex.status).toBe(404);
  });

  test("upstream 400 keeps its status, stream and non-stream", async () => {
    const reject = () => {
      throw new Error('400 {"type":"error","error":{"type":"invalid_request_error","message":"bad"}}');
    };
    faux.setResponses([reject, reject]);
    for (const stream of [false, true]) {
      const response = await call("POST", "/v1/chat/completions", { model: "anthropic/claude-test", messages: [user("hi")], stream });
      expect(response.status).toBe(400);
      expect((await response.json()).error.message).toContain("invalid_request_error");
    }
  });

  test("malformed JSON body is a 400", async () => {
    const url = new URL("http://pi-host/v1/chat/completions");
    const response = await gateway(new Request(url, { method: "POST", body: "{" }), url);
    expect(response?.status).toBe(400);
  });
});

describe("GET /usage", () => {
  const payload = {
    five_hour: { utilization: 12, resets_at: "2026-11-19T18:50:00.182073+00:00" },
    limits: [
      { kind: "session", group: "session", percent: 12, resets_at: "2026-11-19T18:50:00.182073+00:00", scope: null },
      { kind: "weekly_all", group: "weekly", percent: 77, resets_at: "2026-11-20T04:00:00+00:00", scope: null },
      { kind: "weekly_scoped", group: "weekly", percent: 97, resets_at: "2026-11-20T04:00:00+00:00", scope: { model: { display_name: "Fable" } } },
      { kind: "other", group: "monthly", percent: 1, resets_at: null },
    ],
  };

  test("maps limits[] to 5h and 7d windows, model-scoped keeps its label", () => {
    expect(mapAnthropicUsage(payload)).toEqual([
      { window: "5h", usedPercent: 12, resetsAt: "2026-11-19T18:50:00.182Z" },
      { window: "7d", usedPercent: 77, resetsAt: "2026-11-20T04:00:00.000Z" },
      { window: "7d", label: "Fable", usedPercent: 97, resetsAt: "2026-11-20T04:00:00.000Z" },
    ]);
  });

  test("falls back to five_hour/seven_day without limits[]", () => {
    expect(mapAnthropicUsage({ five_hour: { utilization: 3, resets_at: null }, seven_day: null })).toEqual([
      { window: "5h", usedPercent: 3, resetsAt: null },
    ]);
  });

  test("without an OAuth login it is a 503 and never calls Anthropic", async () => {
    const response = await call("GET", "/usage");
    expect(response.status).toBe(503);
    expect(usageCalls).toBe(0);
  });

  test("reports, caches for 60 s, refreshes on ?refresh=1 and keeps upstream 429", async () => {
    // Test seam: tokenFor only reads getAuth, the faux registry has no OAuth login.
    const oauth = { getAuth: async () => ({ source: "OAuth", auth: { apiKey: "token" } }) } as unknown as Models;
    const seenTokens: string[] = [];
    let reply: () => unknown = () => payload;
    const handler = createGateway(
      async () => oauth,
      async (_url, token) => {
        seenTokens.push(token);
        return reply();
      },
    );
    const usage = async (path: string) => (await handler(new Request(`http://pi-host${path}`), new URL(`http://pi-host${path}`)))!;

    const first = await usage("/usage");
    expect(first.status).toBe(200);
    const report = await first.json();
    expect(report.provider).toBe("anthropic");
    expect(report.limits).toHaveLength(3);
    expect(Number.isFinite(Date.parse(report.fetchedAt))).toBe(true);
    await usage("/usage");
    expect(seenTokens).toEqual(["token"]);

    reply = () => {
      throw Object.assign(new Error("HTTP 429"), { status: 429 });
    };
    const limited = await usage("/usage?refresh=1");
    expect(limited.status).toBe(429);
    expect(seenTokens).toHaveLength(2);
  });
});
