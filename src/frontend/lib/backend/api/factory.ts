// Factory routes (`backend/routes/factory.py`, `factory_preview.py`). `repo` is always the path of the primary checkout.
import { toQuery } from "../urls"
import { post, put, request, requestVoid } from "./http"
import type { FactoryPreview, FactoryRepo, FactorySettings, FactoryTask, FactoryTaskCommand, FactoryTaskCreate, FactoryTaskDiff, FactoryTimelineEntry } from "@/lib/domain/factory"

function taskPath(id: string): string {
  return `/factory/tasks/${encodeURIComponent(id)}`
}

export function listFactoryRepos(): Promise<FactoryRepo[]> {
  return request<FactoryRepo[]>("/factory/repos")
}

/** Register the primary git checkout at `path` with the factory, repeating a known path answers the same repo. */
export function openFactoryRepo(path: string): Promise<FactoryRepo> {
  return post<FactoryRepo>("/factory/repos", { path })
}

export function getFactorySettings(repo: string): Promise<FactorySettings> {
  return request<FactorySettings>(`/factory/settings${toQuery({ repo })}`)
}

export function putFactorySettings(repo: string, settings: FactorySettings): Promise<FactorySettings> {
  return put<FactorySettings>(`/factory/settings${toQuery({ repo })}`, settings)
}

/** Tasks of `repo`, newest first. */
export function listFactoryTasks(repo: string): Promise<FactoryTask[]> {
  return request<FactoryTask[]>(`/factory/tasks${toQuery({ repo })}`)
}

export function createFactoryTask(body: FactoryTaskCreate): Promise<FactoryTask> {
  return post<FactoryTask>("/factory/tasks", body)
}

export function getFactoryTask(id: string): Promise<FactoryTask> {
  return request<FactoryTask>(taskPath(id))
}

export function getFactoryTaskDiff(id: string): Promise<FactoryTaskDiff> {
  return request<FactoryTaskDiff>(`${taskPath(id)}/diff`)
}

/** Answers the task as it was just before the command, the new state follows in its session. */
export function commandFactoryTask(id: string, command: FactoryTaskCommand): Promise<FactoryTask> {
  return post<FactoryTask>(`${taskPath(id)}/${command}`, {})
}

/** Make the task's Pi session live (no prompt), so its transcript can be attached to. */
export function openFactoryTaskSession(id: string): Promise<FactoryTask> {
  return post<FactoryTask>(`${taskPath(id)}/session`, {})
}

/** Send `text` to the task's coder, queued as a follow-up while a turn runs. */
export function sendFactoryTaskMessage(id: string, text: string): Promise<FactoryTask> {
  return post<FactoryTask>(`${taskPath(id)}/messages`, { text })
}

/** Refused (409) while the task is running or merging. */
export function deleteFactoryTask(id: string): Promise<void> {
  return requestVoid(taskPath(id), { method: "DELETE" })
}

/** The task's preview, null when none was started. */
export function getFactoryPreview(id: string): Promise<FactoryPreview | null> {
  return request<FactoryPreview | null>(`${taskPath(id)}/preview`)
}

export function startFactoryPreview(id: string): Promise<FactoryPreview> {
  return post<FactoryPreview>(`${taskPath(id)}/preview/start`, {})
}

export function stopFactoryPreview(id: string): Promise<FactoryPreview> {
  return post<FactoryPreview>(`${taskPath(id)}/preview/stop`, {})
}

/** The factory entries of a task's Pi session in append order, only those after the entry `since` when given. */
/** The factory entries of a task's Pi session and its coder's answers, those after the entry `since` when given. */
export async function listFactoryEntries(sessionId: string, since?: string): Promise<FactoryTimelineEntry[]> {
  const { entries } = await request<{ entries: FactoryTimelineEntry[] }>(`/pi/sessions/${encodeURIComponent(sessionId)}/entries${toQuery({ customType: "factory-", since, assistant: "1" })}`)
  return entries
}
