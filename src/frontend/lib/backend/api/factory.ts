// Factory routes: every factory action through the core (`POST /api/factory/call`), and the task previews the backend
// serves itself (`backend/routes/factory_preview.py`).
import { post, request } from "./http"
import type { FactoryActionName, FactoryActions, FactoryPreview } from "@/lib/domain/factory"

/**
 * Run the core's action `action` with `params` and answer its result as is. A refusal throws `ApiError` with the
 * core's one-sentence message, its `code` (`invalid`, `not_found`, `busy`, `refused`, `failed`) and its `detail`
 * (`{force: true}` on a removal `force` would allow).
 */
export function callFactory<T>(action: FactoryActionName, params: Record<string, unknown> = {}): Promise<T> {
  return post<T>("/factory/call", { action, params })
}

/** The core's actions with their parameter schemas. */
export function getFactoryActions(): Promise<FactoryActions> {
  return request<FactoryActions>("/factory/actions")
}

function previewPath(id: string): string {
  return `/factory/tasks/${encodeURIComponent(id)}/preview`
}

/** The task's preview, null when none was started. */
export function getFactoryPreview(id: string): Promise<FactoryPreview | null> {
  return request<FactoryPreview | null>(previewPath(id))
}

export function startFactoryPreview(id: string): Promise<FactoryPreview> {
  return post<FactoryPreview>(`${previewPath(id)}/start`, {})
}

export function stopFactoryPreview(id: string): Promise<FactoryPreview> {
  return post<FactoryPreview>(`${previewPath(id)}/stop`, {})
}
