/**
 * `POST /v1/chat/completions`: OpenAI chat completions over pi-ai, stream and non-stream.
 *
 * - Consumes: a pi-ai `Models` collection, the parsed request body and its `signal` (which aborts the upstream call)
 * - Provides: `chatCompletions`, which keeps the upstream HTTP status of early failures (400, 404, 429)
 */
import type { AssistantMessage, Models } from "@earendil-works/pi-ai";
import { resolveModel } from "./models.ts";
import { HttpError, type OpenAIRequest } from "./openai.ts";
import { streamOptions, toContext } from "./request.ts";
import { completionBody, sseStream } from "./response.ts";

// pi-ai reports upstream HTTP failures only as "<status> <body>" in errorMessage (onResponse does not fire for them).
const upstreamError = (message: AssistantMessage) =>
  new HttpError(Number(/^(\d{3}) /.exec(message.errorMessage ?? "")?.[1] ?? 502), message.errorMessage ?? message.stopReason);

/** Run one chat completion; errors are thrown as `HttpError` for the router to render. */
export async function chatCompletions(models: Models, body: OpenAIRequest, requestSignal: AbortSignal): Promise<Response> {
  const model = await resolveModel(models, body.model);
  // Aborted by the client going away (request.signal) or by the SSE consumer cancelling.
  const abort = new AbortController();
  const signal = AbortSignal.any([requestSignal, abort.signal]);
  const context = await toContext(body, model, signal);
  const options = streamOptions(body, model, signal);
  const id = `chatcmpl-${crypto.randomUUID()}`;
  if (!body.stream) {
    const message = await models.completeSimple(model, context, options);
    if (message.stopReason === "error" || message.stopReason === "aborted") throw upstreamError(message);
    return Response.json(completionBody(message, id, body.model));
  }
  // Upstream request failures arrive as a lone `error` event before `start`: peek so they keep their HTTP status.
  const iterator = models.streamSimple(model, context, options)[Symbol.asyncIterator]();
  const first = await iterator.next();
  if (!first.done && first.value.type === "error") throw upstreamError(first.value.error);
  const events = (async function* () {
    if (!first.done) yield first.value;
    for (let next = await iterator.next(); !next.done; next = await iterator.next()) yield next.value;
  })();
  const stream = sseStream(events, id, body.model, body.stream_options?.include_usage === true, () => abort.abort());
  return new Response(stream, { headers: { "content-type": "text/event-stream", "cache-control": "no-cache" } });
}
