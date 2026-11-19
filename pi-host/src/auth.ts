/** Bearer token and Origin check (M5): loopback is no boundary. */
import { createHash, timingSafeEqual } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { homedir } from "node:os";

export const tokenPath =
  process.env.PI_HOST_TOKEN_FILE ??
  `${process.env.XDG_STATE_HOME ?? `${homedir()}/.local/state`}/gigachad-bot/pi-host.token`;

/** How long a reader waits for a concurrent first start to fill the token file it created. */
const TOKEN_WAIT_ATTEMPTS = 20;
const TOKEN_WAIT_MS = 50;

/**
 * Read the shared token, creating it on first start: exclusively and 0600 from the first byte, so two first starts
 * agree on one token and no other user can read it in between.
 */
export async function loadToken(path = tokenPath): Promise<string> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const token = Buffer.from(crypto.getRandomValues(new Uint8Array(32))).toString("hex");
  try {
    await writeFile(path, `${token}\n`, { mode: 0o600, flag: "wx" });
    return token;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
  }
  for (let attempt = 0; attempt < TOKEN_WAIT_ATTEMPTS; attempt++) {
    const existing = (await readFile(path, "utf8")).trim();
    if (existing) return existing;
    await Bun.sleep(TOKEN_WAIT_MS);
  }
  throw new Error(`token file ${path} is empty, delete it to create a new token`);
}

function digest(value: string): Buffer {
  return createHash("sha256").update(value).digest();
}

/** Reject browser-originated requests and requests without the bearer token (constant-time compare). */
export function authorize(request: Request, token: string): Response | undefined {
  if (request.headers.get("origin")) return new Response("forbidden", { status: 403 });
  const presented = digest(request.headers.get("authorization") ?? "");
  if (!timingSafeEqual(presented, digest(`Bearer ${token}`))) {
    return new Response("unauthorized", { status: 401 });
  }
  return undefined;
}
