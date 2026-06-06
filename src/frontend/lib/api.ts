import type { Attachment, ChatHistoriesResponse, ChatRequest, KanbanCard, Message, ModelsResponse, ProjectData, ProjectListItem, ProjectStateUpdate, ResearchRequest, StudyProcessRequest, StudyProcessResponse, Usage } from "./types"
import { createSSEStream } from "./sse"
import type { SSEStreamResult } from "./sse"
import { API_BASE, DEFAULT_TEMPERATURE, DEFAULT_DOWNSCALE_IMAGES } from "./config"

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    let message = res.statusText
    try {
      const body = await res.json()
      if (body && typeof body === "object" && "detail" in body) {
        message = String(body.detail)
      }
    } catch {
      // not json, use status text
    }
    throw new Error(message)
  }
  return res.json()
}

export async function fetchModels(): Promise<ModelsResponse> {
  return request<ModelsResponse>("/models")
}

export async function fetchPrompts(): Promise<string[]> {
  const data = await request<{ prompts: string[] }>("/prompts")
  return data.prompts
}

export async function fetchHistory(): Promise<{ messages: Message[] }> {
  return request("/history")
}

export async function resetHistory(): Promise<void> {
  await request("/history", { method: "DELETE" })
}

export function createChatStream(req: ChatRequest): SSEStreamResult {
  const body: Record<string, unknown> = {
    model: req.model,
    user_msg: req.user_msg,
    system_prompt: req.system_prompt ?? "",
    temperature: req.temperature ?? DEFAULT_TEMPERATURE,
    downscale_images: req.downscale_images ?? DEFAULT_DOWNSCALE_IMAGES,
    messages: (req.messages ?? []).map((m: Message) => ({
      role: m.role,
      content: m.content,
      tool_call_id: m.tool_call_id,
      tool_calls: m.tool_calls,
    })),
  }
  if (req.reasoning_effort) body.reasoning_effort = req.reasoning_effort
  if (req.img_base64) body.img_base64 = req.img_base64

  return createSSEStream("/chat", body)
}

export async function listChatHistories(): Promise<ChatHistoriesResponse> {
  return request<ChatHistoriesResponse>("/chat-histories")
}

export async function loadChatHistory(filename: string): Promise<{ messages: Message[]; filename: string; chat_id: string | null; title: string | null; usage: Usage | null }> {
  return request(`/chat-histories/${filename}`)
}

export async function saveChatHistory(filename: string, messages: Message[] = [], chatId?: string, title?: string, usage?: Usage): Promise<{ status: string; filename: string }> {
  return request(`/chat-histories/${filename}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, chat_id: chatId ?? null, title: title ?? null, usage: usage ?? null }),
  })
}

export async function deleteChatHistory(filename: string): Promise<void> {
  await request(`/chat-histories/${filename}`, { method: "DELETE" })
}

export async function renameChatHistory(oldPath: string, newTitle: string): Promise<{ status: string; new_path: string; filename: string }> {
  return request(`/chat-histories/rename`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ old_path: oldPath, new_title: newTitle }),
  })
}

export async function archiveChatHistory(filename: string): Promise<void> {
  await request(`/chat-histories/${filename}/archive`, { method: "PUT" })
}

export async function createDirectory(parentPath: string, name: string): Promise<{ status: string; path: string }> {
  return request("/chat-histories/mkdir", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ parent_path: parentPath, name }),
  })
}

export async function moveHistoryItem(filename: string, targetDir: string): Promise<{ status: string; new_path: string }> {
  return request("/chat-histories/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, target_dir: targetDir }),
  })
}

export function createResearchStream(req: ResearchRequest): SSEStreamResult {
  return createSSEStream("/research", {
    query: req.query,
    fast_model: req.fast_model,
    smart_model: req.smart_model,
    strategic_model: req.strategic_model,
    depth: req.depth,
    breadth: req.breadth,
    reasoning_effort: req.reasoning_effort,
    report_type: req.report_type,
  })
}

export function createOCRStream(
  imgBase64: string,
  model: string
): SSEStreamResult {
  return createSSEStream("/ocr", { img_base64: imgBase64, model })
}

function _apiOrigin(): string {
  const url = new URL(API_BASE)
  return url.origin
}

function _uploadsPath(chatId: string, slug: string | null): string {
  const safe = encodeURIComponent(chatId)
  if (slug) return `/chat-histories/${encodeURIComponent(slug)}/_uploads/${safe}`
  return `/chat-uploads/${safe}`
}

export function uploadsBase(chatId: string, slug: string | null): string {
  return `${_apiOrigin()}${_uploadsPath(chatId, slug)}`
}

export function rewriteImages(content: string, chatId: string, slug: string | null): string {
  return content.replace(/\(images\/([^)]+)\)/g, `(${uploadsBase(chatId, slug)}/images/$1)`)
}

export function chatFileUrl(chatId: string, filename: string, slug: string | null = null): string {
  return `${uploadsBase(chatId, slug)}/${encodeURIComponent(filename)}`
}

export async function uploadFile(chatId: string, file: File, slug: string | null = null): Promise<Attachment> {
  const form = new FormData()
  form.append("file", file)
  const params = new URLSearchParams({ chat_id: chatId })
  if (slug) params.set("slug", slug)
  const res = await fetch(`${API_BASE}/files/upload?${params}`, { method: "POST", body: form })
  if (!res.ok) throw new Error(await res.text())
  const data = await res.json()
  return { ...data, url: chatFileUrl(chatId, data.name, slug) }
}

export interface ParsedAttachment {
  name: string
  parsedMd: string | null
}

export async function parseFiles(chatId: string, filenames: string[], slug: string | null = null): Promise<ParsedAttachment[]> {
  const params = new URLSearchParams({ chat_id: chatId })
  for (const f of filenames) params.append("filenames", f)
  if (slug) params.set("slug", slug)
  const res = await fetch(`${API_BASE}/files/parse?${params}`, { method: "POST" })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function deleteAttachment(chatId: string, filename: string, slug: string | null = null): Promise<void> {
  const params = slug ? `?slug=${encodeURIComponent(slug)}` : ""
  await request(`/files/chat/${chatId}/att/${encodeURIComponent(filename)}${params}`, { method: "DELETE" })
}

export async function deleteChatUploads(chatId: string, slug: string | null = null): Promise<void> {
  const params = slug ? `?slug=${encodeURIComponent(slug)}` : ""
  await request(`/files/chat/${chatId}${params}`, { method: "DELETE" })
}

export async function processStudyPdf(req: StudyProcessRequest): Promise<StudyProcessResponse> {
  return request<StudyProcessResponse>("/study/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  })
}

export async function listProjects(): Promise<ProjectListItem[]> {
  const data = await request<{ projects: ProjectListItem[] }>("/projects")
  return data.projects
}

export async function loadProject(name: string): Promise<ProjectData> {
  return request<ProjectData>(`/projects/${encodeURIComponent(name)}`)
}

export async function createProject(name: string): Promise<ProjectData> {
  return request<ProjectData>("/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  })
}

export async function deleteProject(name: string): Promise<void> {
  await request(`/projects/${encodeURIComponent(name)}`, { method: "DELETE" })
}

export async function updateProjectKanban(name: string, data: ProjectStateUpdate): Promise<ProjectData> {
  return request<ProjectData>(`/projects/${encodeURIComponent(name)}/state`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
}

export async function addProjectCard(name: string, title: string, description: string = "", state: string = "backlog"): Promise<KanbanCard> {
  return request<KanbanCard>(`/projects/${encodeURIComponent(name)}/cards`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, description, state }),
  })
}

export async function moveProjectCard(name: string, cardId: string, state: string): Promise<KanbanCard> {
  return request<KanbanCard>(`/projects/${encodeURIComponent(name)}/cards/${cardId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state }),
  })
}

export async function updateProjectCard(name: string, cardId: string, title?: string, description?: string): Promise<KanbanCard> {
  return request<KanbanCard>(`/projects/${encodeURIComponent(name)}/cards/${cardId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, ...(description !== undefined ? { description } : {}) }),
  })
}

export async function deleteProjectCard(name: string, cardId: string): Promise<void> {
  await request(`/projects/${encodeURIComponent(name)}/cards/${cardId}`, { method: "DELETE" })
}

export async function saveProjectTab(name: string, filename: string, messages: Message[], chatId?: string, tabName?: string, title?: string, usage?: Usage): Promise<{ status: string }> {
  return request(`/projects/${encodeURIComponent(name)}/tabs/${encodeURIComponent(filename)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, messages, chat_id: chatId ?? null, tab_name: tabName ?? null, title: title ?? null, usage: usage ?? null }),
  })
}

export async function deleteProjectTab(name: string, filename: string): Promise<void> {
  await request(`/projects/${encodeURIComponent(name)}/tabs/${encodeURIComponent(filename)}`, { method: "DELETE" })
}

export async function loadProjectTab(name: string, filename: string): Promise<{ messages: Message[]; filename: string; chat_id: string | null; title: string | null; usage: Usage | null }> {
  return request(`/chat-histories/${encodeURIComponent(name)}/${encodeURIComponent(filename)}`)
}

export abstract class Entry {
  constructor(public readonly id: string) {}
  abstract delete(): Promise<void>
}

export class Element extends Entry {
  async delete(): Promise<void> {
    await request(`/chat-histories/${this.id}`, { method: "DELETE" })
  }
}

export class Vault extends Entry {
  async delete(): Promise<void> {
    await request(`/chat-histories/${this.id}`, { method: "DELETE" })
  }
}

export class Directory extends Vault {}

export class Project extends Vault {
  async delete(): Promise<void> {
    await request(`/projects/${encodeURIComponent(this.id)}`, { method: "DELETE" })
  }
}