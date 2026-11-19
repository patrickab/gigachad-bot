/** Bearer token and Origin check (M5): loopback is no boundary. */
import { chmod, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { homedir } from "node:os";

export const tokenPath =
  process.env.PI_HOST_TOKEN_FILE ??
  `${process.env.XDG_STATE_HOME ?? `${homedir()}/.local/state`}/gigachad-bot/pi-host.token`;

/** Read the shared token, creating it (0600) on first start. */
export async function loadToken(): Promise<string> {
  const file = Bun.file(tokenPath);
  if (await file.exists()) {
    const existing = (await file.text()).trim();
    if (existing) return existing;
  }
  await mkdir(dirname(tokenPath), { recursive: true, mode: 0o700 });
  const token = Buffer.from(crypto.getRandomValues(new Uint8Array(32))).toString("hex");
  await Bun.write(tokenPath, `${token}\n`);
  await chmod(tokenPath, 0o600);
  return token;
}

/** Reject browser-originated requests and requests without the bearer token. */
export function authorize(request: Request, token: string): Response | undefined {
  if (request.headers.get("origin")) return new Response("forbidden", { status: 403 });
  if (request.headers.get("authorization") !== `Bearer ${token}`) {
    return new Response("unauthorized", { status: 401 });
  }
  return undefined;
}
