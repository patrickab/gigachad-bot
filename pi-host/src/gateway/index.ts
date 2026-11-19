/**
 * Completion gateway and quota: `GET /v1/models`, `POST /v1/chat/completions`, `GET /usage` (plan "Gateway details").
 *
 * - Consumes: Pi's `ModelRuntime` (created once, lazily, see `runtime.ts` for what it may write)
 * - Provides: `handleGateway(request, url)` for `server.ts` (undefined for other routes), `createGateway` for tests
 */
import type { Models } from "@earendil-works/pi-ai";
import { chatCompletions } from "./completions.ts";
import { modelsBody } from "./models.ts";
import { errorResponse } from "./openai.ts";
import { createPiRuntime } from "./runtime.ts";
import { createUsage, type FetchJson } from "./usage.ts";

export type GatewayHandler = (request: Request, url: URL) => Promise<Response | undefined>;

/** The gateway routes over `models`, resolved on first use. */
export function createGateway(models: () => Promise<Models>, fetchJson?: FetchJson): GatewayHandler {
  const usage = createUsage(models, fetchJson);
  return async (request, url) => {
    const route = `${request.method} ${url.pathname}`;
    try {
      if (route === "GET /v1/models") return Response.json(await modelsBody(await models()));
      if (route === "POST /v1/chat/completions") return await chatCompletions(await models(), request);
      if (route === "GET /usage") return await usage(url.searchParams.get("refresh") === "1");
      return undefined;
    } catch (error) {
      return errorResponse(error);
    }
  };
}

let runtime: Promise<Models> | undefined;
const piRuntime = () => {
  runtime ??= createPiRuntime().catch((error) => {
    runtime = undefined; // Retry on the next request instead of caching a failed start.
    throw error;
  });
  return runtime;
};

/** Completion gateway (/v1/models, /v1/chat/completions) and /usage over Pi's runtime. */
export const handleGateway: GatewayHandler = createGateway(piRuntime);
