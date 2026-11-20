/**
 * Pi extension pi-host passes with `-e` to its local `coding` and `document` children: the provider `gateway`, this
 * pi-host's completion gateway (`/v1`), for models of providers the child has no credentials for.
 *
 * - Consumes: `PI_GATEWAY_URL` (pi-host's `/v1`), `PI_GATEWAY_TOKEN` (the gateway-only token, C1), `GET /v1/models`
 * - Provides: provider `gateway`, model ids `<provider>/<id>` as sandboxed children name them (C2)
 *
 * Provider keys stay in pi-host. Models of providers the child already runs itself (the Anthropic login, `ollama`)
 * are dropped once the session starts, so each model has one route. Without the env (a user's own Pi) it does nothing.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type GatewayModel = {
  id: string;
  owned_by: string;
  name?: string;
  context_window?: number;
  max_tokens?: number;
  input?: ("text" | "image")[];
  reasoning?: boolean;
};

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

function register(pi: ExtensionAPI, baseUrl: string, models: GatewayModel[]): void {
  pi.registerProvider("gateway", {
    name: "pi-host gateway",
    baseUrl,
    apiKey: "$PI_GATEWAY_TOKEN",
    api: "openai-completions",
    models: models.map((model) => ({
      id: model.id,
      name: model.name ?? model.id,
      reasoning: model.reasoning ?? false,
      input: model.input ?? ["text"],
      cost: ZERO_COST,
      contextWindow: model.context_window ?? 128000,
      maxTokens: model.max_tokens ?? 16384,
    })),
  });
}

export default async function (pi: ExtensionAPI) {
  const baseUrl = process.env.PI_GATEWAY_URL;
  const token = process.env.PI_GATEWAY_TOKEN;
  if (!baseUrl || !token) return;
  let models: GatewayModel[];
  try {
    const response = await fetch(`${baseUrl}/models`, { headers: { authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(10_000) });
    if (!response.ok) return;
    const payload: unknown = await response.json();
    if (!payload || typeof payload !== "object" || !("data" in payload) || !Array.isArray(payload.data)) return;
    // pi-host's own `modelsBody` shape (src/gateway/models.ts).
    models = payload.data as GatewayModel[];
  } catch {
    return;
  }
  // All of them at load, so a resumed session on a gateway model restores it.
  register(pi, baseUrl, models);
  pi.on("session_start", (_event, ctx) => {
    const own = new Set(ctx.modelRegistry.getAvailable().map((model) => model.provider).filter((provider) => provider !== "gateway"));
    register(pi, baseUrl, models.filter((model) => !own.has(model.owned_by)));
  });
}
