// Learn session routes.
import { toQuery } from "../urls"
import { jsonInit, patch, request, requestVoid } from "./http"
import type { LearnSession } from "@/lib/domain/types"

/** Every learn session with the project it is grouped under. Host-access logins only. */
export async function fetchLearnSessions(signal?: AbortSignal): Promise<LearnSession[]> {
  return request<LearnSession[]>("/learn/sessions", { signal })
}

/** Group a learn session under `projectSlug` (out of any other project), or under none. */
export async function assignLearnSession(sessionId: string, projectSlug: string | null): Promise<void> {
  await requestVoid(`/learn/sessions/${encodeURIComponent(sessionId)}/project`, await jsonInit("PUT", { project_slug: projectSlug }))
}

/** Rename a learn session in LearnHarness, resolving to the title it stored. */
export async function renameLearnSession(sessionId: string, title: string): Promise<string> {
  return (await patch<{ title: string }>(`/learn/sessions/${encodeURIComponent(sessionId)}`, { title })).title
}

/** How a learn session is deleted: its Knowledge stays in the graph, or (LearnHarness's "Also remove its graph traces") goes too. */
export type LearnDeleteMode = "preserve-graph" | "delete-nodes"

/** Delete a learn session in LearnHarness and drop it from its project. */
export async function deleteLearnSession(sessionId: string, mode: LearnDeleteMode = "preserve-graph"): Promise<void> {
  await requestVoid(`/learn/sessions/${encodeURIComponent(sessionId)}${toQuery({ mode })}`, { method: "DELETE" })
}

/** A learn session as LearnHarness exports it: one Markdown document and the file name it suggests. */
export async function exportLearnSession(sessionId: string): Promise<{ filename: string; markdown: string }> {
  return request<{ filename: string; markdown: string }>(`/learn/sessions/${encodeURIComponent(sessionId)}/markdown`)
}
