// Canvas snapshot and mutation routes.
import { encodePath, toQuery } from "../urls"
import { post, request } from "./http"
import type { CanvasMutation, CanvasSnapshot } from "@/lib/domain/canvasDocument"

/** Annotation canvas key of a PDF, shared by every copy with the same bytes. Creates the empty canvas on first use. */
export async function openPdfAnnotations(path: string): Promise<string> {
  const data = await request<{ key: string }>(`/fileviewer/annotations${toQuery({ path })}`, { method: "POST" })
  return data.key
}

/** API path of one canvas's snapshot, `/mutations` and `/stream` routes: the owner's by key, or a share link's. */
export type CanvasEndpoint = string & { readonly __canvasEndpoint: unique symbol }

export function canvasEndpoint(canvasKey: string): CanvasEndpoint {
  return `/canvases/${encodePath(canvasKey)}` as CanvasEndpoint
}

/** Canvas snapshots carry their durable mutation revision for gapless stream resume. */
export async function loadCanvasSnapshot(endpoint: CanvasEndpoint): Promise<CanvasSnapshot> {
  return request<CanvasSnapshot>(endpoint)
}

/** Resolves with the revision holding these mutations; the stream delivers their content. */
export async function submitCanvasMutations(endpoint: CanvasEndpoint, mutations: CanvasMutation[]): Promise<{ revision: number }> {
  return post<{ revision: number }>(`${endpoint}/mutations`, { mutations })
}
