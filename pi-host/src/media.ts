/**
 * Content-addressed store for tool-result images, so frames and history carry a hash instead of inline base64.
 *
 * - Consumes: Pi records and entries (parsed JSON), `PI_HOST_MEDIA_DIR` (default `$XDG_CACHE_HOME/gigachad-bot/pi-media`)
 * - Provides: `withMediaHashes` (image blocks of tool results become `{type:"image", mimeType, hash}`), `handleMedia`
 *   (`GET /media/<sha256>`: the bytes, `ETag` the hash, `immutable`), `mediaBytes`
 * - The hash is the sha256 of the decoded bytes. Only raster images that sniff as PNG, JPEG, GIF or WebP are stored,
 *   anything else stays inline so no image is ever dropped. Thumbnails are the backend's job (`lib/pi_media.py`)
 * - Files older than 30 days are swept daily, the next attach stores a session's images again
 * - `forgetMedia` drops images of a deleted turn, from disk and from the map of recently stored ones
 */
import { createHash } from "node:crypto";
import { mkdirSync, readdirSync, renameSync, rmSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const HASH_RE = /^[a-f0-9]{64}$/;
const SWEEP_EVERY_MS = 24 * 60 * 60 * 1000;
const MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;
/** Live tool output resends the whole result on each update: remember which base64 strings were stored already. */
const RECENT_MAX_CHARS = 16 * 1024 * 1024;

export type MediaImage = { type: "image"; mimeType: string; hash: string };

let dirOverride: string | undefined;
let sweptAt = 0;
const recent = new Map<string, string>();
let recentChars = 0;

/** Test hook: store in `dir` (undefined restores the default) and forget what was stored. */
export function useMediaDir(dir: string | undefined): void {
  dirOverride = dir;
  sweptAt = 0;
  recent.clear();
  recentChars = 0;
}

function mediaDir(): string {
  return dirOverride ?? process.env.PI_HOST_MEDIA_DIR ?? join(process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"), "gigachad-bot", "pi-media");
}

/** Raster image type from magic bytes, null for anything else. */
export function sniffImage(bytes: Uint8Array): string | null {
  const at = (index: number) => bytes[index];
  const text = (start: number, end: number) => String.fromCharCode(...bytes.subarray(start, end));
  if (at(0) === 0x89 && text(1, 4) === "PNG") return "image/png";
  if (at(0) === 0xff && at(1) === 0xd8 && at(2) === 0xff) return "image/jpeg";
  if (text(0, 4) === "GIF8") return "image/gif";
  if (text(0, 4) === "RIFF" && text(8, 12) === "WEBP") return "image/webp";
  return null;
}

function sweep(): void {
  const now = Date.now();
  if (now - sweptAt < SWEEP_EVERY_MS) return;
  sweptAt = now;
  let names: string[];
  try {
    names = readdirSync(mediaDir());
  } catch {
    return;
  }
  for (const name of names) {
    const file = join(mediaDir(), name);
    try {
      if (now - statSync(file).mtimeMs > MAX_AGE_MS) unlinkSync(file);
    } catch {
      // Removed concurrently: the next sweep retries.
    }
  }
}

/** Store a base64 image under its content address; undefined when it is not a raster image or the disk fails. */
function store(base64: string): string | undefined {
  const known = recent.get(base64);
  const bytes = Buffer.from(base64, "base64");
  const hash = known ?? createHash("sha256").update(bytes).digest("hex");
  if (!sniffImage(bytes)) return undefined;
  const file = join(mediaDir(), hash);
  if (statSync(file, { throwIfNoEntry: false })?.size !== bytes.length) {
    sweep();
    const tmp = `${file}.${crypto.randomUUID()}.tmp`;
    try {
      mkdirSync(mediaDir(), { recursive: true });
      writeFileSync(tmp, bytes);
      renameSync(tmp, file);
    } catch (error) {
      rmSync(tmp, { force: true });
      console.warn(`pi-host: cannot store media ${hash.slice(0, 12)}: ${String(error)}`);
      return undefined;
    }
  }
  if (!known && base64.length <= RECENT_MAX_CHARS) {
    recent.set(base64, hash);
    recentChars += base64.length;
    while (recentChars > RECENT_MAX_CHARS) {
      const oldest = recent.keys().next().value!;
      recentChars -= oldest.length;
      recent.delete(oldest);
    }
  }
  return hash;
}

/**
 * Forget the base64 of `hashes`, then remove their stored images, each on its own, so nothing of them stays in memory
 * and as little as possible on disk. Answers the hashes whose file could not be removed.
 */
export function forgetMedia(hashes: readonly string[]): string[] {
  const gone = new Set(hashes.filter((hash) => HASH_RE.test(hash)));
  for (const [base64, hash] of recent) {
    if (!gone.has(hash)) continue;
    recent.delete(base64);
    recentChars -= base64.length;
  }
  const left: string[] = [];
  for (const hash of gone) {
    try {
      rmSync(join(mediaDir(), hash), { force: true });
    } catch {
      left.push(hash);
    }
  }
  return left;
}

type Json = Record<string, unknown>;

function isRecord(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function mediaBlock(block: unknown): unknown {
  if (!isRecord(block) || block.type !== "image" || typeof block.data !== "string" || block.data === "") return block;
  const hash = store(block.data);
  if (!hash) return block;
  return { type: "image", mimeType: typeof block.mimeType === "string" ? block.mimeType : "image/png", hash };
}

/**
 * `value` with the image blocks of every tool result replaced by media blocks, the same object when it has none. A tool
 * result is a message of role `toolResult` or the `result` / `partialResult` of a `tool_execution_*` record. Images of
 * user prompts and custom messages stay as they are.
 */
export function withMediaHashes<T>(value: T): T {
  const walk = (node: unknown, inResult: boolean): unknown => {
    if (Array.isArray(node)) {
      const mapped = node.map((item) => (inResult ? mediaBlock(walk(item, true)) : walk(item, false)));
      return mapped.some((item, index) => item !== node[index]) ? mapped : node;
    }
    if (!isRecord(node)) return node;
    const toolExecution = typeof node.type === "string" && node.type.startsWith("tool_execution");
    let changed: Json | undefined;
    for (const [key, child] of Object.entries(node)) {
      const into = inResult || node.role === "toolResult" || (toolExecution && (key === "result" || key === "partialResult"));
      const next = walk(child, into);
      if (next !== child) (changed ??= { ...node })[key] = next;
    }
    return changed ?? node;
  };
  return walk(value, false) as T;
}

/** The stored image's bytes and type, undefined for an unknown hash. */
export async function mediaBytes(hash: string): Promise<{ bytes: Uint8Array<ArrayBuffer>; mimeType: string } | undefined> {
  if (!HASH_RE.test(hash)) return undefined;
  const file = Bun.file(join(mediaDir(), hash));
  if (!(await file.exists())) return undefined;
  const bytes = new Uint8Array(await file.arrayBuffer());
  const mimeType = sniffImage(bytes);
  return mimeType ? { bytes, mimeType } : undefined;
}

/** `GET /media/<sha256>`: the image with its hash as `ETag`, cacheable forever. Undefined for other routes. */
export async function handleMedia(request: Request, url: URL): Promise<Response | undefined> {
  const match = /^\/media\/(.+)$/.exec(url.pathname);
  if (!match) return undefined;
  if (request.method !== "GET") return new Response("method not allowed", { status: 405 });
  const found = await mediaBytes(match[1]);
  if (!found) return new Response("not found", { status: 404 });
  const etag = `"${match[1]}"`;
  const headers = { ETag: etag, "Cache-Control": "public, max-age=31536000, immutable" };
  if (request.headers.get("if-none-match") === etag) return new Response(null, { status: 304, headers });
  return new Response(found.bytes, { headers: { ...headers, "Content-Type": found.mimeType } });
}
