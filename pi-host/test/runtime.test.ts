/** Pi's runtime over a scratch agent dir: a `models.json` custom provider (Ollama) is listed, served and reloaded on change. */
import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Server } from "bun";
import { createGateway, type GatewayHandler } from "../src/gateway/index.ts";
import { reloadingPiRuntime } from "../src/gateway/runtime.ts";

type HttpServer = Server<undefined>;

const savedAgentDir = process.env.PI_CODING_AGENT_DIR;
let scratch: string;
let agentDir: string;
let gateway: GatewayHandler;
let ollama: HttpServer;
let upstreamModels: string[];

/** A fake Ollama `/v1/chat/completions` answering every request with one streamed sentence. */
function fakeOllama(): HttpServer {
  return Bun.serve({
    port: 0,
    async fetch(request) {
      const body = (await request.json()) as { model: string };
      upstreamModels.push(body.model);
      const chunk = (delta: object, finish: string | null) =>
        `data: ${JSON.stringify({ id: "c1", object: "chat.completion.chunk", model: body.model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
      const usage = `data: ${JSON.stringify({ id: "c1", object: "chat.completion.chunk", model: body.model, choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } })}\n\n`;
      const sse = chunk({ role: "assistant", content: "hello from " }, null) + chunk({ content: body.model }, "stop") + usage + "data: [DONE]\n\n";
      return new Response(sse, { headers: { "content-type": "text/event-stream" } });
    },
  });
}

/** Replace `models.json` in one rename, as the backend's sync does, with an `ollama` provider of `ids`. */
async function writeModelsJson(...ids: string[]): Promise<void> {
  const provider = {
    baseUrl: `http://127.0.0.1:${ollama.port}/v1`,
    api: "openai-completions",
    apiKey: "ollama",
    models: ids.map((id) => ({ id, input: ["text"], reasoning: false, contextWindow: 8192 })),
  };
  const staging = join(agentDir, ".models.json.tmp");
  await writeFile(staging, JSON.stringify({ providers: { ollama: provider } }));
  await rename(staging, join(agentDir, "models.json"));
}

async function call(method: string, path: string, body?: unknown): Promise<Response> {
  const url = new URL(`http://pi-host${path}`);
  const init = body === undefined ? { method } : { method, body: JSON.stringify(body), headers: { "content-type": "application/json" } };
  const response = await gateway(new Request(url, init), url);
  if (!response) throw new Error(`no route for ${method} ${path}`);
  return response;
}

async function ollamaIds(): Promise<string[]> {
  const body = (await (await call("GET", "/v1/models")).json()) as { data: { id: string }[] };
  return body.data.map((model) => model.id).filter((id) => id.startsWith("ollama/"));
}

beforeAll(async () => {
  scratch = await mkdtemp(join(tmpdir(), "pi-host-runtime-"));
  ollama = fakeOllama();
});

beforeEach(async () => {
  agentDir = join(scratch, crypto.randomUUID());
  await mkdir(agentDir, { recursive: true });
  process.env.PI_CODING_AGENT_DIR = agentDir;
  upstreamModels = [];
  gateway = createGateway(reloadingPiRuntime());
});

afterAll(async () => {
  ollama.stop(true);
  await rm(scratch, { recursive: true, force: true });
  if (savedAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = savedAgentDir;
});

describe("models.json custom provider", () => {
  test("is listed and served through the gateway", async () => {
    await writeModelsJson("qwen3:0.6b");
    expect(await ollamaIds()).toEqual(["ollama/qwen3:0.6b"]);

    const response = await call("POST", "/v1/chat/completions", { model: "ollama/qwen3:0.6b", messages: [{ role: "user", content: "hi" }] });
    expect(response.status).toBe(200);
    const body = (await response.json()) as { model: string; choices: { message: { content: string } }[] };
    expect(body.model).toBe("ollama/qwen3:0.6b");
    expect(body.choices[0]!.message.content).toBe("hello from qwen3:0.6b");
    expect(upstreamModels).toEqual(["qwen3:0.6b"]);
  });

  test("changes show up without a restart", async () => {
    await writeModelsJson("qwen3:0.6b");
    expect(await ollamaIds()).toEqual(["ollama/qwen3:0.6b"]);

    await writeModelsJson("qwen3:0.6b", "llava:7b");
    expect(await ollamaIds()).toEqual(["ollama/qwen3:0.6b", "ollama/llava:7b"]);

    await writeModelsJson("llava:7b");
    expect(await ollamaIds()).toEqual(["ollama/llava:7b"]);
    const removed = await call("POST", "/v1/chat/completions", { model: "ollama/qwen3:0.6b", messages: [{ role: "user", content: "hi" }] });
    expect(removed.status).toBe(404);
  });

  test("a models.json created after the start is picked up", async () => {
    expect(await ollamaIds()).toEqual([]);
    await writeModelsJson("qwen3:0.6b");
    expect(await ollamaIds()).toEqual(["ollama/qwen3:0.6b"]);
  });
});
