// Fetch core shared by every route family: base URL, device headers, status checks, JSON and SSE bodies.
import { getApiBase } from "../config"
import { deviceHeaders } from "../deviceId"
import { createSSEStream } from "../sse"
import type { SSEStreamResult } from "../sse"

/**
 * Carries the HTTP status so callers can tell a stale-revision rejection from a real failure, and the `code` and
 * `detail` of a `{detail: {code, message, detail?}}` body (the factory core's refusals).
 */
export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly code?: string, readonly detail?: Record<string, unknown>) {
    super(message)
    this.name = "ApiError"
  }
}

/** Throws ApiError, preferring FastAPI's `detail` (string, validation list or `{code, message}`) over the status text. */
async function ensureOk(res: Response): Promise<Response> {
  if (!res.ok) {
    let message = res.statusText
    let code: string | undefined
    let extra: Record<string, unknown> | undefined
    try {
      const body = await res.json()
      if (body && typeof body === "object" && "detail" in body) {
        const detail = body.detail
        if (Array.isArray(detail)) {
          message = detail.map((d: { msg?: string }) => d.msg || String(d)).join("; ")
        } else if (typeof detail === "string") {
          message = detail
        } else if (detail && typeof detail === "object" && typeof detail.message === "string") {
          message = detail.message
          if (typeof detail.code === "string") code = detail.code
          if (detail.detail && typeof detail.detail === "object" && !Array.isArray(detail.detail)) extra = detail.detail
        } else {
          message = String(detail)
        }
      }
    } catch {
      // not json, use status text
    }
    throw new ApiError(message, res.status, code, extra)
  }
  return res
}

// Chrome pools uncredentialed and credentialed requests on separate connections. Tailscale Serve can drop an
// HTTP/2 connection without a reset, and the browser reuses it until TCP gives up, so a stuck caller switches pools.
let connectionPool = 0

export function currentConnectionPool(): number {
  return connectionPool
}

/** Send later requests over a fresh connection, once per stuck pool however many callers report it. */
export function leaveConnectionPool(pool: number): void {
  if (pool === connectionPool) connectionPool += 1
}

/** Methods a repeat cannot apply twice, so a request the backend never answered is safe to send again. */
const IDEMPOTENT_METHODS = new Set(["GET", "HEAD", "PUT", "DELETE", "OPTIONS"])
/** How long a request waits out an unreachable backend (a hot reload or restart) before it fails. */
const OFFLINE_RETRY_WINDOW_MS = 30_000
const OFFLINE_RETRY_MIN_MS = 250
const OFFLINE_RETRY_MAX_MS = 2_000

/** `retryOffline` sends a POST again too when the backend was unreachable: only for one that is safe to repeat. */
export type ApiInit = RequestInit & { retryOffline?: boolean }

function sleep(ms: number, signal?: AbortSignal | null): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(done, ms)
    signal?.addEventListener("abort", done, { once: true })
    function done() {
      clearTimeout(timer)
      signal?.removeEventListener("abort", done)
      resolve()
    }
  })
}

/**
 * The single place that adds the API base URL and the device headers, the status is left to the caller.
 * A network failure (fetch's TypeError: the backend is restarting or reloading) is retried with backoff for up to
 * `OFFLINE_RETRY_WINDOW_MS` when the method is idempotent or `retryOffline` is set. An aborted signal ends it at once.
 */
async function apiFetch(path: string, init: ApiInit = {}): Promise<Response> {
  const { retryOffline, ...fetchInit } = init
  const retries = retryOffline || IDEMPOTENT_METHODS.has((fetchInit.method ?? "GET").toUpperCase())
  const deadline = Date.now() + OFFLINE_RETRY_WINDOW_MS
  for (let delay = OFFLINE_RETRY_MIN_MS; ; delay = Math.min(delay * 2, OFFLINE_RETRY_MAX_MS)) {
    const headers = { ...(fetchInit.headers as Record<string, string> | undefined), ...deviceHeaders() }
    const pool = connectionPool
    const credentials: RequestCredentials = pool % 2 ? "include" : "same-origin"
    try {
      return await fetch(`${getApiBase()}${path}`, { credentials, ...fetchInit, headers })
    } catch (error) {
      fetchInit.signal?.throwIfAborted()
      // A connection that failed is dead or stale (Firefox: "NetworkError when attempting to fetch resource"): the retry takes a fresh one
      if (error instanceof TypeError) leaveConnectionPool(pool)
      if (!retries || !(error instanceof TypeError) || Date.now() + delay > deadline) throw error
    }
    await sleep(delay, fetchInit.signal)
    fetchInit.signal?.throwIfAborted()
  }
}

export async function request<T>(path: string, options?: ApiInit): Promise<T> {
  const res = await ensureOk(await apiFetch(path, options))
  return res.json()
}

/** Like `request` for endpoints whose response body is not read. */
export async function requestVoid(path: string, init?: ApiInit): Promise<void> {
  await ensureOk(await apiFetch(path, init))
}

/** Bodies from this size on are gzipped, smaller ones are not worth the round of compression. */
const GZIP_REQUEST_MIN_BYTES = 64 * 1024

/** `body` as a JSON request, gzipped when large: a long chat's save resends every message, the backend inflates it (`TransferMiddleware`). */
export async function jsonInit(method: string, body: unknown, headers: Record<string, string> = {}): Promise<RequestInit> {
  const json = JSON.stringify(body)
  const init = { method, headers: { "Content-Type": "application/json", ...headers }, body: json as BodyInit }
  if (json.length < GZIP_REQUEST_MIN_BYTES || typeof CompressionStream === "undefined") return init
  const gzipped = await new Response(new Blob([json]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer()
  return { ...init, headers: { ...init.headers, "Content-Encoding": "gzip" }, body: gzipped }
}

/** POSTs `body` as JSON and streams the SSE response. */
export function sseStream(path: string, body: Record<string, unknown>): SSEStreamResult {
  // A chat request the backend never answered is sent again: at worst a reply is generated twice, never lost to a stale connection
  return createSSEStream(async (signal) => apiFetch(path, { ...await jsonInit("POST", body), signal, retryOffline: true }))
}

/** `retryOffline`: the POST is safe to repeat, so it waits out an unreachable backend like an idempotent request. */
export async function post<T>(path: string, body: unknown, options: { retryOffline?: boolean } = {}): Promise<T> {
  return request<T>(path, { ...await jsonInit("POST", body), ...options })
}

export async function put<T>(path: string, body: unknown, headers: Record<string, string> = {}): Promise<T> {
  return request<T>(path, await jsonInit("PUT", body, headers))
}

export async function patch<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, await jsonInit("PATCH", body))
}

export function del<T>(path: string): Promise<T> {
  // keepalive: a deletion committed while the page unloads still reaches the backend
  return request<T>(path, { method: "DELETE", keepalive: true })
}

export function fileForm(file: File | Blob, filename?: string): FormData {
  const form = new FormData()
  if (filename !== undefined) form.append("file", file, filename)
  else form.append("file", file)
  return form
}
