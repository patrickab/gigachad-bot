/**
 * Model listing and lookup, limited to D13's providers.
 *
 * - Consumes: a pi-ai `Models` collection (Pi's `ModelRuntime` in production)
 * - Provides: `listedProvider` (D13 filter), `resolveModel` (404 on unknown or unlisted), `modelsBody` (`GET /v1/models`)
 */
import { type Api, getSupportedThinkingLevels, type Model, type Models } from "@earendil-works/pi-ai";
import { HttpError } from "./openai.ts";

/** D13: Anthropic (subscription) plus providers configured by API key, never Codex. */
export async function listedProvider(models: Models, providerId: string): Promise<boolean> {
  if (providerId === "openai-codex") return false;
  if (providerId === "anthropic") return true;
  return (await models.checkAuth(providerId))?.type === "api_key";
}

/** Look up `<provider>/<model>` (or a bare model id, first match) among the listed providers. */
export async function resolveModel(models: Models, id: unknown): Promise<Model<Api>> {
  if (typeof id !== "string" || !id) throw new HttpError(400, "model is required");
  const slash = id.indexOf("/");
  const model =
    slash > 0
      ? models.getModel(id.slice(0, slash), id.slice(slash + 1))
      : models.getModels().find((candidate) => candidate.id === id);
  if (!model || !(await listedProvider(models, model.provider))) throw new HttpError(404, `unknown model ${id}`);
  return model;
}

/** The OpenAI model list with each model's Pi thinking ladder. */
export async function modelsBody(models: Models) {
  const available = await models.getAvailable();
  const providers = [...new Set(available.map((model) => model.provider))];
  const listed = new Set(
    (await Promise.all(providers.map(async (id) => ((await listedProvider(models, id)) ? id : undefined)))).filter(Boolean),
  );
  return {
    object: "list",
    data: available
      .filter((model) => listed.has(model.provider))
      .map((model) => ({
        id: `${model.provider}/${model.id}`,
        object: "model",
        created: 0,
        owned_by: model.provider,
        name: model.name,
        context_window: model.contextWindow,
        max_tokens: model.maxTokens,
        input: model.input,
        reasoning: model.reasoning,
        thinking_levels: model.reasoning ? getSupportedThinkingLevels(model) : [],
      })),
  };
}
