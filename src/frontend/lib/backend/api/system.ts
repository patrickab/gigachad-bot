// Backend config, provider usage, OCR stream and redeploy routes.
import type { SSEStreamResult } from "../sse"
import { toQuery } from "../urls"
import { jsonInit, request, requestVoid, sseStream } from "./http"
import type { BackendConfig, ProviderUsageReport } from "@/lib/domain/types"

/** Subscription usage per Pi login; `refresh` skips the server's 5 minute cache. */
export async function fetchProviderUsage(refresh = false): Promise<{ reports: ProviderUsageReport[] }> {
  return request<{ reports: ProviderUsageReport[] }>(`/provider-usage${toQuery({ refresh })}`)
}

export async function fetchBackendConfig(signal?: AbortSignal): Promise<BackendConfig> {
  return request<BackendConfig>("/config", { signal })
}

/** Store whether the animated app background shows, for this user. */
export async function saveAnimatedBackground(enabled: boolean): Promise<void> {
  await requestVoid("/config/animated-background", await jsonInit("PUT", { enabled }))
}

export function createOCRStream(
  imgBase64: string,
  model?: string
): SSEStreamResult {
  return sseStream("/ocr", { img_base64: imgBase64, model })
}

/** Start the production redeploy, the backend restarts shortly after. Host-access logins only. */
export async function startRedeploy(): Promise<void> {
  await requestVoid("/deploy/redeploy", { method: "POST" })
}
