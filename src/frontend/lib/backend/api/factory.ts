// Factory engine routes.
import { toQuery } from "../urls"
import { post, put, request } from "./http"
import type { FactoryDefaults, FactoryEvent, FactoryMessage, FactoryPreview, FactoryRepo, FactoryTask, FactoryTaskDetail, FactoryWorkflow, TaskAction, TaskCreate } from "@/lib/domain/factory"

export function fetchLatestFactoryEventSeq(): Promise<number> {
  return request<{ seq: number }>("/factory/events/latest").then((cursor) => cursor.seq)
}

export function fetchFactoryRepos(): Promise<FactoryRepo[]> {
  return request<FactoryRepo[]>("/factory/repos")
}

/** Register the git checkout at *path* with the factory engine, repeating a known path returns the same repo. */
export function registerFactoryRepo(path: string): Promise<FactoryRepo> {
  return post<FactoryRepo>("/factory/repos", { path })
}

export function fetchFactoryDefaults(): Promise<FactoryDefaults> {
  return request<FactoryDefaults>("/factory/defaults")
}

export function saveFactoryDefaults(defaults: FactoryDefaults): Promise<FactoryDefaults> {
  return put<FactoryDefaults>("/factory/defaults", defaults)
}

export function fetchFactoryWorkflows(repo: string): Promise<FactoryWorkflow[]> {
  return request<FactoryWorkflow[]>(`/factory/workflows${toQuery({ repo })}`)
}

export function fetchFactoryTasks(repo: string): Promise<FactoryTask[]> {
  return request<FactoryTask[]>(`/factory/tasks${toQuery({ repo })}`)
}

export function fetchFactoryTask(id: string): Promise<FactoryTaskDetail> {
  return request<FactoryTaskDetail>(`/factory/tasks/${encodeURIComponent(id)}`)
}

export function fetchFactoryEvents(repo: string, after: number): Promise<FactoryEvent[]> {
  return request<FactoryEvent[]>(`/factory/events${toQuery({ repo, after })}`)
}

export function createFactoryTask(body: TaskCreate): Promise<FactoryTask> {
  return post<FactoryTask>("/factory/tasks", body)
}

export function runFactoryTaskAction(id: string, action: TaskAction): Promise<FactoryTask> {
  return post<FactoryTask>(`/factory/tasks/${encodeURIComponent(id)}/${action}`, {})
}

export function sendFactoryMessage(id: string, text: string): Promise<FactoryMessage> {
  return post<FactoryMessage>(`/factory/tasks/${encodeURIComponent(id)}/messages`, { text })
}

export function restartFactoryPreview(id: string): Promise<FactoryPreview> {
  return post<FactoryPreview>(`/factory/tasks/${encodeURIComponent(id)}/preview/restart`, {})
}
