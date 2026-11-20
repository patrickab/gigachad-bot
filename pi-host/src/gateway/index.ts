/**
 * Completion gateway and quota: `GET /v1/models`, `POST /v1/chat/completions`, `GET /usage` (plan "Gateway details").
 *
 * - Consumes: Pi's `ModelRuntime` (created once, lazily, reloading `models.json`, see `runtime.ts` for what it may write),
 *   optionally the OMP gateway (`omp.ts`), which takes the Anthropic chat completions
 * - Provides: `handleGateway(request, url)` for `server.ts` (undefined for other routes), `createGateway` for tests
 */
import type { Models } from "@earendil-works/pi-ai";
import { chatCompletions } from "./completions.ts";
import { modelsBody } from "./models.ts";
import { type OmpGateway, ompGateway } from "./omp.ts";
import { errorResponse } from "./openai.ts";
import { reloadingPiRuntime } from "./runtime.ts";
import { createUsage, type FetchJson } from "./usage.ts";

export type GatewayHandler = (request: Request, url: URL) => Promise<Response | undefined>;

/** The `model` of a chat completion body, undefined when the body is not JSON (the Pi path answers 400). */
function bodyModel(body: string): unknown {
  try {
    return JSON.parse(body)?.model;
  } catch {
    return undefined;
  }
}

/** The gateway routes over `models`, resolved on first use; `omp` takes the chat completions it handles. */
export function createGateway(models: () => Promise<Models>, fetchJson?: FetchJson, omp?: OmpGateway): GatewayHandler {
  const usage = createUsage(models, fetchJson);
  return async (request, url) => {
    const route = `${request.method} ${url.pathname}`;
    try {
      if (route === "GET /v1/models") return Response.json(await modelsBody(await models()));
      if (route === "POST /v1/chat/completions") {
        if (omp) {
          const body = await request.clone().text();
          if (omp.handles(bodyModel(body))) return await omp.forward(request, body);
        }
        return await chatCompletions(await models(), request);
      }
      if (route === "GET /usage") return await usage(url.searchParams.get("refresh") === "1");
      return undefined;
    } catch (error) {
      return errorResponse(error);
    }
  };
}

/** Completion gateway (/v1/models, /v1/chat/completions) and /usage over Pi's runtime, Anthropic completions through OMP when installed. */
export const handleGateway: GatewayHandler = createGateway(reloadingPiRuntime(), undefined, ompGateway());
