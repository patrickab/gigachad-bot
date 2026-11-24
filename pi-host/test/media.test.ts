/** Content-addressed tool-result images: hashing, what is rewritten, the cached route. */
import { afterAll, beforeEach, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { handleMedia, sniffImage, useMediaDir, withMediaHashes } from "../src/media.ts";

const dir = mkdtempSync(join(tmpdir(), "pi-host-media-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
beforeEach(() => useMediaDir(join(dir, crypto.randomUUID())));

// 1x1 PNG
const PNG = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", "base64");
const BASE64 = PNG.toString("base64");
const HASH = createHash("sha256").update(PNG).digest("hex");
const image = { type: "image", data: BASE64, mimeType: "image/png" };
const toolResult = { role: "toolResult", toolCallId: "t1", content: [{ type: "text", text: "ok" }, image] };

describe("withMediaHashes", () => {
  test("a tool result's image becomes its sha256 and the bytes are stored", async () => {
    const out = withMediaHashes(toolResult) as { content: unknown[] };
    expect(out.content[1]).toEqual({ type: "image", mimeType: "image/png", hash: HASH });
    expect(out.content[0]).toBe(toolResult.content[0]);
    const response = (await handleMedia(new Request(`http://x/media/${HASH}`), new URL(`http://x/media/${HASH}`)))!;
    expect(Buffer.from(await response.arrayBuffer()).equals(PNG)).toBe(true);
  });

  test("covers tool_execution results, message_end, entries and agent_end", () => {
    const records = [
      { type: "tool_execution_update", toolCallId: "t1", partialResult: { content: [image] } },
      { type: "tool_execution_end", toolCallId: "t1", result: { content: [image] } },
      { type: "message_end", message: toolResult },
      { type: "message", id: "e1", message: toolResult },
      { type: "agent_end", messages: [toolResult] },
    ];
    for (const record of records) expect(JSON.stringify(withMediaHashes(record))).not.toContain(BASE64);
  });

  test("leaves user prompt images, other records and non-images alone", () => {
    const user = { role: "user", content: [image] };
    const start = { type: "tool_execution_start", args: { content: [image] } };
    const text = { role: "toolResult", content: [{ type: "text", text: "x" }] };
    expect(withMediaHashes(user)).toBe(user);
    expect(withMediaHashes(start)).toBe(start);
    expect(withMediaHashes(text)).toBe(text);
  });

  test("an image that is not a raster image stays inline, never dropped", () => {
    const odd = { role: "toolResult", content: [{ type: "image", data: Buffer.from("<svg/>").toString("base64"), mimeType: "image/svg+xml" }] };
    expect(withMediaHashes(odd)).toBe(odd);
  });

  test("the same image twice is one file", () => {
    const a = withMediaHashes(toolResult) as { content: { hash?: string }[] };
    const b = withMediaHashes({ type: "tool_execution_end", result: { content: [image] } }) as { result: { content: { hash?: string }[] } };
    expect(a.content[1].hash).toBe(b.result.content[0].hash);
    expect(sniffImage(PNG)).toBe("image/png");
  });
});

describe("GET /media/<hash>", () => {
  const get = (hash: string, headers: Record<string, string> = {}, method = "GET") =>
    handleMedia(new Request(`http://x/media/${hash}`, { method, headers }), new URL(`http://x/media/${hash}`)) as Promise<Response>;

  test("serves the bytes with an ETag and immutable caching, 304 on a repeat", async () => {
    withMediaHashes(toolResult);
    const response = await get(HASH);
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toBe("image/png");
    expect(response.headers.get("etag")).toBe(`"${HASH}"`);
    expect(response.headers.get("cache-control")).toBe("public, max-age=31536000, immutable");
    const repeat = await get(HASH, { "if-none-match": `"${HASH}"` });
    expect(repeat.status).toBe(304);
  });

  test("unknown or malformed hashes are 404, other routes are not handled", async () => {
    expect((await get(HASH)).status).toBe(404);
    expect((await get("..%2F..%2Fetc%2Fpasswd")).status).toBe(404);
    expect((await get(HASH, {}, "POST")).status).toBe(405);
    expect(await handleMedia(new Request("http://x/files"), new URL("http://x/files"))).toBeUndefined();
  });
});
