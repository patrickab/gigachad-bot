// Fetch core shared by every route family: base URL, device headers, status checks, JSON and SSE bodies.
import { getApiBase } from "../config"
import { deviceHeaders } from "../deviceId"
import { createSSEStream } from "../sse"
import type { SSEStreamResult } from "../sse"

/** Carries the HTTP status so callers can tell a stale-revision rejection from a real failure. */
export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = "ApiError"
  }
}

/** Throws ApiError, preferring FastAPI's `detail` (string or validation list) over the status text. */
async function ensureOk(res: Response): Promise<Response> {
  if (!res.ok) {
    let message = res.statusText
    try {
      const body = await res.json()
      if (body && typeof body === "object" && "detail" in body) {
        const detail = body.detail
        if (Array.isArray(detail)) {
          message = detail.map((d: { msg?: string }) => d.msg || String(d)).join("; ")
        } else if (typeof detail === "string") {
          message = detail
        } else {
          message = String(detail)
        }
      }
    } catch {
      // not json, use status text
    }
    throw new ApiError(message, res.status)
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

/** The single place that adds the API base URL and the device headers, the status is left to the caller. */
function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = { ...(init?.headers as Record<string, string> | undefined), ...deviceHeaders() }
  const credentials: RequestCredentials = connectionPool % 2 ? "include" : "same-origin"
  return fetch(`${getApiBase()}${path}`, { credentials, ...init, headers })
}

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await ensureOk(await apiFetch(path, options))
  return res.json()
}

/** Like `request` for endpoints whose response body is not read. */
export async function requestVoid(path: string, init?: RequestInit): Promise<void> {
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
  return createSSEStream(async (signal) => apiFetch(path, { ...await jsonInit("POST", body), signal }))
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, await jsonInit("POST", body))
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
