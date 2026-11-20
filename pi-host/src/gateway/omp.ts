/**
 * OMP gateway: Anthropic chat completions through `omp auth-gateway`, the path that worked before the Pi migration.
 *
 * - Consumes: the `omp` binary, its credential store (`$PI_CONFIG_DIR` or `~/.omp`: `agent/agent.db`, `auth-gateway.token`),
 *   `PI_HOST_OMP_GATEWAY=0` (opt out), `PI_HOST_OMP_BROKER_BIND` (default 127.0.0.1:8765), `PI_HOST_OMP_GATEWAY_BIND` (default 127.0.0.1:4000)
 * - Provides: `ompGateway()` (undefined when OMP is not installed or opted out) with `handles(model)` and `forward(request, body)`;
 *   `stopOmp()` ends the services this process started
 *
 * Anthropic models only: OMP shapes subscription requests as its own client, which `pi-ai` alone does not.
 * Both OMP services start on first use and a service already listening on its port is reused, never restarted.
 * `/v1/models` stays Pi's, OMP serves the same `anthropic/<id>` model ids.
 */
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type OmpGateway = {
  handles(model: unknown): boolean;
  forward(request: Request, body: string): Promise<Response>;
};

const STARTUP_TIMEOUT_MS = 120_000;
const owned: Bun.Subprocess[] = [];

async function isListening(bind: string): Promise<boolean> {
  const [hostname, port] = bind.split(":");
  try {
    const socket = await Bun.connect({ hostname, port: Number(port), socket: { data() {} } });
    socket.end();
    return true;
  } catch {
    return false;
  }
}

/** Start `omp <args>` unless `bind` already answers, and wait until it does or the child exits. */
async function ensureService(label: string, bind: string, args: string[], env: Record<string, string>): Promise<void> {
  if (await isListening(bind)) return;
  const child = Bun.spawn(["omp", ...args, `--bind=${bind}`], { env: { ...process.env, ...env }, stdout: "inherit", stderr: "inherit" });
  owned.push(child);
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (await isListening(bind)) return;
    if (child.exitCode !== null) {
      if (await isListening(bind)) return; // A racing instance won the port.
      throw new Error(`omp ${label} exited during startup (code ${child.exitCode})`);
    }
    await Bun.sleep(250);
  }
  child.kill();
  throw new Error(`omp ${label} did not answer on ${bind}`);
}

async function ompOutput(args: string[]): Promise<string> {
  const child = Bun.spawn(["omp", ...args], { stdout: "pipe", stderr: "inherit" });
  const output = await new Response(child.stdout).text();
  if ((await child.exited) !== 0) throw new Error(`omp ${args.join(" ")} failed`);
  return output.trim().split("\n").at(-1) ?? "";
}

/** Stop the OMP services this process started. */
export function stopOmp(): void {
  for (const child of owned.splice(0)) child.kill("SIGTERM");
}

/** The OMP gateway, or undefined when `omp` or its credential store is missing or `PI_HOST_OMP_GATEWAY=0`. */
export function ompGateway(): OmpGateway | undefined {
  const home = process.env.PI_CONFIG_DIR ?? join(homedir(), ".omp");
  if (process.env.PI_HOST_OMP_GATEWAY === "0" || !Bun.which("omp") || !existsSync(join(home, "agent", "agent.db"))) return undefined;
  const brokerBind = process.env.PI_HOST_OMP_BROKER_BIND ?? "127.0.0.1:8765";
  const gatewayBind = process.env.PI_HOST_OMP_GATEWAY_BIND ?? "127.0.0.1:4000";

  let started: Promise<string> | undefined;
  /** Both services running, resolved to the gateway bearer. A failed start retries on the next call. */
  const start = () =>
    (started ??= (async () => {
      await ensureService("auth-broker", brokerBind, ["auth-broker", "serve"], {});
      const env = { OMP_AUTH_BROKER_URL: `http://${brokerBind}`, OMP_AUTH_BROKER_TOKEN: await ompOutput(["auth-broker", "token"]) };
      await ensureService("auth-gateway", gatewayBind, ["auth-gateway", "serve"], env);
      return (await Bun.file(join(home, "auth-gateway.token")).text()).trim();
    })().catch((error) => {
      started = undefined;
      throw error;
    }));

  return {
    handles: (model) => typeof model === "string" && model.startsWith("anthropic/"),
    async forward(request, body) {
      const token = await start();
      const upstream = await fetch(`http://${gatewayBind}/v1/chat/completions`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
        body,
        signal: request.signal,
      });
      const headers = new Headers({ "content-type": upstream.headers.get("content-type") ?? "application/json" });
      if (upstream.headers.has("cache-control")) headers.set("cache-control", upstream.headers.get("cache-control")!);
      return new Response(upstream.body, { status: upstream.status, headers });
    },
  };
}
