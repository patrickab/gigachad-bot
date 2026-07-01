import type { Attachment, BranchMeta, CategoryDef, ChatHistoriesResponse, ChatRequest, KanbanCard, MemoryExtractResponse, MemoryPreviewResponse, Message, ModelsResponse, ObsidianFile, ObsidianNode, PreviewMemory, ProjectData, ProjectDocument, ProjectListItem, ProjectStateUpdate, ProposedMemory, ResearchRequest, StudyProcessRequest, StudyProcessResponse, Usage } from "./types"
import { createSSEStream } from "./sse"
import type { SSEStreamResult } from "./sse"
import { API_BASE, DEFAULT_TEMPERATURE, DEFAULT_DOWNSCALE_IMAGES } from "./config"

function encodePath(filename: string): string {
  return filename.split("/").map(encodeURIComponent).join("/")
}

export function parseHistoryFile(historyFile: string): { slug: string | null; filename: string } {
  const parts = historyFile.split("/")
  if (parts.length > 1) {
    return { slug: parts[0], filename: parts.slice(1).join("/") }
  }
  return { slug: null, filename: historyFile }
}

export function buildHistoryFile(filename: string, slug: string | null): string {
  return slug ? `${slug}/${filename}` : filename
}

async function ensureOk(res: Response): Promise<Response> {
  if (!res.ok) {
    let message = res.statusText
    try {
      const body = await res.json()
      if (body && typeof body === "object" && "detail" in body) {
        const detail = body.detail
        if (Array.isArray(detail)) {
          message = detail.map((d: any) => d.msg || String(d)).join("; ")
        } else if (typeof detail === "string") {
          message = detail
        } else {
          message = String(detail)
        }
      }
    } catch {
      // not json, use status text
    }
    throw new Error(message)
  }
  return res
}

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await ensureOk(await fetch(`${API_BASE}${path}`, options))
  return res.json()
}

export async function fetchModels(): Promise<ModelsResponse> {
  return request<ModelsResponse>("/models")
}

export async function fetchPrompts(): Promise<Record<string, string>> {
  const data = await request<{ prompts: Record<string, string> }>("/prompts")
  return data.prompts
}

export interface PromptMeta { slug: string; name: string; includes: string[] }

export async function fetchPromptList(): Promise<PromptMeta[]> {
  return request<PromptMeta[]>("/prompts/list")
}

export async function fetchPromptBlocks(): Promise<Record<string, string>> {
  return request<Record<string, string>>("/prompts/blocks")
}

export async function fetchPromptRaw(slug: string): Promise<{ slug: string; content: string }> {
  return request<{ slug: string; content: string }>(`/prompts/${encodeURIComponent(slug)}`)
}

export async function savePrompt(slug: string, content: string): Promise<{ slug: string; name: string }> {
  return request<{ slug: string; name: string }>(`/prompts/${encodeURIComponent(slug)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  })
}

export async function deletePrompt(slug: string): Promise<void> {
  await request<{ deleted: boolean }>(`/prompts/${encodeURIComponent(slug)}`, { method: "DELETE" })
}

export async function savePromptOrder(order: string[]): Promise<void> {
  await request<{ ok: boolean }>("/prompt-order", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order }),
  })
}

export function createChatStream(req: ChatRequest): SSEStreamResult {
  const body: Record<string, unknown> = {
    model: req.model,
    chat_id: req.chat_id,
    user_msg: req.user_msg,
    system_prompt: req.system_prompt ?? "",
    temperature: req.temperature ?? DEFAULT_TEMPERATURE,
    downscale_images: req.downscale_images ?? DEFAULT_DOWNSCALE_IMAGES,
    img_paths: req.img_paths ?? [],
    messages: req.messages ?? [],
    project_slug: req.project_slug ?? null,
  }
  if (req.reasoning_effort) body.reasoning_effort = req.reasoning_effort

  return createSSEStream("/chat", body)
}

export async function listChatHistories(): Promise<ChatHistoriesResponse> {
  return request<ChatHistoriesResponse>("/chat-histories")
}

export async function fetchBranchMeta(dirs?: string[]): Promise<Record<string, BranchMeta>> {
  const params = dirs && dirs.length > 0 ? `?dir=${dirs.map(encodeURIComponent).join(",")}` : ""
  return request(`/chat-histories/branch-meta${params}`)
}

export async function loadChatHistory(filename: string): Promise<{ messages: Message[]; filename: string; chat_id: string | null; title: string | null; usage: Usage | null; parent_id: string | null; branch_message_idx: number | null; children: BranchMeta["children"] }> {
  return request(`/chat-histories/${filename}`)
}

export async function saveChatHistory(filename: string, messages: Message[] = [], chatId?: string, title?: string, usage?: Usage, parentId?: string | null, branchMessageIdx?: number | null, children?: BranchMeta["children"] | null): Promise<{ status: string; filename: string }> {
  const body: Record<string, unknown> = { messages, chat_id: chatId ?? null, title: title ?? null, usage: usage ?? null }
  if (parentId !== undefined) body.parent_id = parentId
  if (branchMessageIdx !== undefined) body.branch_message_idx = branchMessageIdx
  if (children !== undefined) body.children = children
  return request(`/chat-histories/${filename}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
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

export async function createBranch(parentFile: string, branchMessageIdx: number): Promise<{ status: string; child_file: string; chat_id: string }> {
  return request("/chat-histories/branch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ parent_file: parentFile, branch_message_idx: branchMessageIdx }),
  })
}

export async function mergeBranch(childFile: string): Promise<{ status: string }> {
  return request(`/chat-histories/merge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ child_file: childFile }),
  })
}

export async function cascadeDelete(filename: string): Promise<{ status: string; deleted: string[] }> {
  return request(`/chat-histories/cascade/${encodePath(filename)}`, {
    method: "DELETE",
  })
}

export async function orphanChildren(filename: string): Promise<{ status: string; orphaned: string[] }> {
  return request(`/chat-histories/orphan/${encodePath(filename)}`, {
    method: "DELETE",
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

function uploadsBase(chatId: string, slug: string | null): string {
  return `${_apiOrigin()}${_uploadsPath(chatId, slug)}`
}

export function rewriteImages(content: string, chatId: string, slug: string | null): string {
  return content.replace(/\(images\/([^)]+)\)/g, `(${uploadsBase(chatId, slug)}/images/$1)`)
}

export function chatFileUrl(chatId: string, filename: string, slug: string | null = null): string {
  return `${uploadsBase(chatId, slug)}/${encodeURIComponent(filename)}`
}

export async function uploadFile(chatId: string, file: File, slug: string | null = null, overwrite = false): Promise<Attachment> {
  const form = new FormData()
  form.append("file", file)
  const params = new URLSearchParams({ chat_id: chatId })
  if (slug) params.set("slug", slug)
  if (overwrite) params.set("overwrite", "true")
  const res = await ensureOk(await fetch(`${API_BASE}/files/upload?${params}`, { method: "POST", body: form }))
  const data = await res.json()
  return { ...data, active: true, url: chatFileUrl(chatId, data.name, slug) }
}

export interface ParsedAttachment {
  name: string
  parsedMd: string | null
}

export async function parseFiles(chatId: string, filenames: string[], slug: string | null = null): Promise<ParsedAttachment[]> {
  const params = new URLSearchParams({ chat_id: chatId })
  for (const f of filenames) params.append("filenames", f)
  if (slug) params.set("slug", slug)
  const res = await ensureOk(await fetch(`${API_BASE}/files/parse?${params}`, { method: "POST" }))
  return res.json()
}

export async function deleteAttachment(chatId: string, filename: string, slug: string | null = null): Promise<void> {
  const params = slug ? `?slug=${encodeURIComponent(slug)}` : ""
  await request(`/files/chat/${chatId}/att/${encodeURIComponent(filename)}${params}`, { method: "DELETE" })
}

export async function listObsidianFiles(): Promise<{ enabled: boolean; files: ObsidianFile[] }> {
  return request("/obsidian/files")
}

export async function obsidianTree(): Promise<{ enabled: boolean; tree: ObsidianNode[] }> {
  return request("/obsidian/tree")
}

export async function addObsidianRoot(path: string): Promise<{ enabled: boolean; tree: ObsidianNode[] }> {
  return request("/obsidian/roots", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  })
}

export async function removeObsidianRoot(path: string): Promise<{ enabled: boolean; tree: ObsidianNode[] }> {
  return request(`/obsidian/roots?path=${encodeURIComponent(path)}`, { method: "DELETE" })
}

export async function addObsidianMountpoint(
  vault: string,
  path: string,
): Promise<{ enabled: boolean; tree: ObsidianNode[] }> {
  return request(`/obsidian/mountpoints?vault=${encodeURIComponent(vault)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  })
}

export async function removeObsidianMountpoint(
  vault: string,
  path: string,
): Promise<{ enabled: boolean; tree: ObsidianNode[] }> {
  return request(
    `/obsidian/mountpoints?vault=${encodeURIComponent(vault)}&path=${encodeURIComponent(path)}`,
    { method: "DELETE" },
  )
}

export async function readObsidianRendered(path: string): Promise<string> {
  const data = await request<{ path: string; content: string }>(`/obsidian/rendered?path=${encodeURIComponent(path)}`)
  return data.content
}

export async function writeObsidianFile(path: string, content: string): Promise<void> {
  await request("/obsidian/file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, content }),
  })
}

/** Direct URL to a file's raw bytes (images, PDFs) — usable as an `<img>`/PDF src. */
export function fileViewerRawUrl(path: string): string {
  return `${API_BASE}/fileviewer/raw?path=${encodeURIComponent(path)}`
}

/** Read a file's text content (markdown / unknown-treated-as-text). */
export async function loadFileViewerText(path: string): Promise<string> {
  const data = await request<{ path: string; content: string }>(`/fileviewer/text?path=${encodeURIComponent(path)}`)
  return data.content
}

/** Materialise a file at *path* into the chat's upload dir as an Attachment. */
async function attachFileByPath(endpoint: string, chatId: string, path: string, slug: string | null): Promise<Attachment> {
  const params = new URLSearchParams({ path, chat_id: chatId })
  if (slug) params.set("slug", slug)
  const res = await ensureOk(await fetch(`${API_BASE}/${endpoint}/attach?${params}`, { method: "POST" }))
  const data = await res.json()
  return { ...data, active: true, url: chatFileUrl(chatId, data.name, slug) }
}

export function attachObsidianFile(chatId: string, path: string, slug: string | null = null): Promise<Attachment> {
  return attachFileByPath("obsidian", chatId, path, slug)
}

export async function listProjectDocuments(slug: string): Promise<ProjectDocument[]> {
  const data = await request<{ documents: ProjectDocument[] }>(`/documents?slug=${encodeURIComponent(slug)}`)
  return data.documents
}

export async function listAllDocuments(): Promise<ProjectDocument[]> {
  const data = await request<{ documents: ProjectDocument[] }>("/documents/all")
  return data.documents
}

export async function uploadDocument(slug: string, file: File): Promise<ProjectDocument> {
  const form = new FormData()
  form.append("file", file)
  const res = await ensureOk(await fetch(`${API_BASE}/documents/upload?slug=${encodeURIComponent(slug)}`, { method: "POST", body: form }))
  return res.json()
}

export async function registerUploadAsDocument(chatId: string, filename: string, slug: string | null = null): Promise<void> {
  const params = new URLSearchParams()
  if (slug) params.set("slug", slug)
  await request(`/documents/register-upload?${params}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, filename }),
  })
}

export async function addDocument(slug: string, path: string): Promise<ProjectDocument> {
  return request<ProjectDocument>(`/documents/add?slug=${encodeURIComponent(slug)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  })
}

export async function removeDocument(slug: string, path: string): Promise<void> {
  const params = new URLSearchParams({ slug, path })
  await request(`/documents?${params}`, { method: "DELETE" })
}

export function attachDocument(chatId: string, path: string, slug: string | null = null): Promise<Attachment> {
  return attachFileByPath("documents", chatId, path, slug)
}

export async function writeDocument(slug: string, name: string, content: string = ""): Promise<ProjectDocument> {
  return request<ProjectDocument>("/documents/write", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slug, name, content }),
  })
}

export async function writeBinaryDocument(slug: string, filename: string, blob: Blob): Promise<ProjectDocument> {
  const form = new FormData()
  form.append("file", blob, filename)
  return request<ProjectDocument>(`/documents/write-binary?slug=${encodeURIComponent(slug)}`, {
    method: "POST",
    body: form,
  })
}

/** Mirror a rendered canvas (.jpg) into the browsable cloud Drawings collection. */
export async function mirrorDrawing(filename: string, blob: Blob): Promise<void> {
  const form = new FormData()
  form.append("file", blob, filename)
  await request("/documents/mirror-drawing", { method: "POST", body: form })
}

export async function generateMindmap(messages: { role: string; content: string }[], model: string, prompt: string = ""): Promise<string> {
  const data = await request<{ mindmap: string }>("/study/mindmap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, model, prompt }),
  })
  return data.mindmap
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

export async function loadProjectTab(name: string, filename: string): Promise<{ messages: Message[]; filename: string; chat_id: string | null; title: string | null; usage: Usage | null; parent_id: string | null; branch_message_idx: number | null }> {
  return request(`/chat-histories/${encodeURIComponent(name)}/${encodeURIComponent(filename)}`)
}

export class Vault {
  constructor(public readonly id: string) {}
  async delete(): Promise<void> {
    await request(`/chat-histories/${this.id}`, { method: "DELETE" })
  }
}

// --- Memory API ---

function memoryRequest<T>(path: string, body: Record<string, unknown>): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

export async function extractMemories(
  messages: { role: string; content: string }[],
  projectSlug?: string | null,
  chatId?: string | null,
  scope?: "global" | "project" | null,
): Promise<MemoryExtractResponse> {
  return memoryRequest<MemoryExtractResponse>("/memory/extract", {
    messages,
    project_slug: projectSlug ?? null,
    chat_id: chatId ?? null,
    scope: scope ?? null,
  })
}

export async function cancelMemories(reviewId: string, projectSlug?: string | null): Promise<void> {
  await memoryRequest("/memory/cancel", { review_id: reviewId, project_slug: projectSlug ?? null })
}

export async function commitMemoryDoc(
  scope: string,
  acceptedMemories: ProposedMemory[],
  projectSlug: string | null,
  reviewId: string | null,
  rejectedMemories: ProposedMemory[] | null,
  revisedMemories?: PreviewMemory[] | null,
): Promise<void> {
  await memoryRequest("/memory/commit", {
    scope,
    accepted_memories: acceptedMemories,
    project_slug: projectSlug,
    review_id: reviewId,
    rejected_memories: rejectedMemories,
    ...(revisedMemories ? { revised_memories: revisedMemories } : {}),
  })
}

export async function previewMemoryDoc(
  scope: string,
  acceptedMemories: ProposedMemory[],
  projectSlug: string | null,
): Promise<MemoryPreviewResponse> {
  return memoryRequest("/memory/preview", {
    scope,
    accepted_memories: acceptedMemories,
    project_slug: projectSlug,
  })
}

export async function getMemories(scope: "global" | "project", projectSlug?: string | null): Promise<{ memories: PreviewMemory[] }> {
  const params = new URLSearchParams({ scope })
  if (projectSlug) params.set("project_slug", projectSlug)
  return request<{ memories: PreviewMemory[] }>(`/memory/memories?${params.toString()}`)
}

export async function getCategories(scope: "global" | "project", projectSlug?: string | null): Promise<{ categories: CategoryDef[] }> {
  const params = new URLSearchParams({ scope })
  if (projectSlug) params.set("project_slug", projectSlug)
  return request<{ categories: CategoryDef[] }>(`/memory/categories?${params.toString()}`)
}

export async function saveCategories(scope: "global" | "project", categories: CategoryDef[], projectSlug?: string | null): Promise<void> {
  await request("/memory/categories", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scope, categories, project_slug: projectSlug ?? null }),
  })
}

export async function moveMemory(
  memoryId: string,
  fromScope: "global" | "project",
  toScope: "global" | "project",
  fromProjectSlug?: string | null,
  toProjectSlug?: string | null,
): Promise<void> {
  await request("/memory/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      memory_id: memoryId,
      from_scope: fromScope,
      to_scope: toScope,
      from_project_slug: fromProjectSlug ?? null,
      to_project_slug: toProjectSlug ?? null,
    }),
  })
}

export async function remapOrphanedCategory(
  scope: "global" | "project",
  orphanedMemories: PreviewMemory[],
  remainingCategories: CategoryDef[],
  projectSlug?: string | null,
): Promise<{ memories: PreviewMemory[] }> {
  return request("/memory/remap-category", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scope,
      orphaned_memories: orphanedMemories,
      remaining_categories: remainingCategories,
      project_slug: projectSlug ?? null,
    }),
  })
}
