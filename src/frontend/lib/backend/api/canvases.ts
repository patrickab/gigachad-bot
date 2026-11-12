// Canvas snapshot and mutation routes.
import { encodePath, toQuery } from "../urls"
import { post, request } from "./http"
import type { CanvasMutation, CanvasSnapshot } from "@/lib/domain/canvasDocument"

/** Annotation canvas key of a PDF, shared by every copy with the same bytes. Creates the empty canvas on first use. */
export async function openPdfAnnotations(path: string): Promise<string> {
  const data = await request<{ key: string }>(`/fileviewer/annotations${toQuery({ path })}`, { method: "POST" })
  return data.key
}

/** Canvas snapshots carry their durable mutation revision for gapless stream resume. */
export async function loadCanvasSnapshot(canvasKey: string): Promise<CanvasSnapshot> {
  return request<CanvasSnapshot>(`/canvases/${encodePath(canvasKey)}`)
}

/** Resolves with the revision holding these mutations; the stream delivers their content. */
export async function submitCanvasMutations(canvasKey: string, mutations: CanvasMutation[]): Promise<{ revision: number }> {
  return post<{ revision: number }>(`/canvases/${encodePath(canvasKey)}/mutations`, { mutations })
}
