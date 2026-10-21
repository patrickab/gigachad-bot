import type { Attachment, BackendConfig, BranchMeta, CategoryDef, ChatHistoriesResponse, ChatRequest, DirectoryListing, KanbanCard, MemoryExtractResponse, MemoryPreviewResponse, Message, ModelDefaults, ModelProvider, ModelsResponse, OmpArchivedSession, OmpCatalog, OmpOverview, OmpSystemPrompt, OmpWorkspace, FactoryDefaults, FactoryEvent, FactoryMessage, FactoryPreview, FactoryRepo, FactoryTask, FactoryTaskDetail, FactoryWorkflow, TaskAction, TaskCreate, PreviewMemory, ProjectData, ProjectDocument, ProjectListItem, ProjectStateUpdate, ProposedMemory, ReasoningLevels, Usage, RootFile, FileTreeNode } from "@/lib/domain/types"
import { createSSEStream } from "./sse"
import type { SSEStreamResult } from "./sse"
import { getApiBase } from "./config"
import type { ArchitectureOperation, ArchitectureWorkspace } from "@/lib/domain/architecture"
import type { CanvasMutation, CanvasSnapshot } from "@/lib/domain/canvasDocument"
import { deviceHeaders } from "./deviceId"
import { chatFileUrl, encodePath, fileViewerRawUrl, toQuery } from "./urls"
import { baseName, extOf, kindOf } from "@/lib/paths"
import { architectureSlug, architectureSource, isArchitecturePath } from "@/lib/domain/architecturePaths"

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

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = { ...(options?.headers as Record<string, string> | undefined), ...deviceHeaders() }
  const res = await ensureOk(await fetch(`${getApiBase()}${path}`, { ...options, headers }))
  return res.json()
}

function jsonInit(method: string, body: unknown, headers: Record<string, string> = {}): RequestInit {
  return { method, headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) }
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, jsonInit("POST", body))
}

function put<T>(path: string, body: unknown, headers: Record<string, string> = {}): Promise<T> {
  return request<T>(path, jsonInit("PUT", body, headers))
}

function patch<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, jsonInit("PATCH", body))
}

function del<T>(path: string): Promise<T> {
  return request<T>(path, { method: "DELETE" })
}

function fileForm(file: File | Blob, filename?: string): FormData {
  const form = new FormData()
  if (filename !== undefined) form.append("file", file, filename)
  else form.append("file", file)
  return form
}

export async function fetchModels(): Promise<ModelsResponse> {
  return request<ModelsResponse>("/models")
}

export interface ProviderUsageWindow {
  id: "5h" | "7d" | "30d"
  percent: number
  resets_in_minutes: number | null
}

export interface ProviderUsageReport {
  provider: string
  account: string | null
  no_limits: boolean
  /** omp could not read this account's usage this time, e.g. a rate-limited usage API. */
  unavailable?: boolean
  windows: ProviderUsageWindow[]
}

/** Subscription usage per OMP login; `refresh` skips the server's 5 minute cache. */
export async function fetchProviderUsage(refresh = false): Promise<{ reports: ProviderUsageReport[] }> {
  return request<{ reports: ProviderUsageReport[] }>(`/provider-usage${refresh ? "?refresh=true" : ""}`)
}

export async function fetchReasoningLevels(model: string): Promise<ReasoningLevels> {
  return request<ReasoningLevels>(`/models/reasoning-levels?model=${encodeURIComponent(model)}`)
}

export async function saveModelProviders(providers: ModelProvider[]): Promise<ModelsResponse> {
  return put<ModelsResponse>("/models/providers", {
    providers: Object.fromEntries(
      providers.map(({ label, litellm_id, models, source }) => [label, { litellm_id, models, source: source ?? null }]),
    ),
  })
}

export async function saveModelTabOrder(order: string[]): Promise<ModelsResponse> {
  return put<ModelsResponse>("/models/tab-order", { order })
}

export async function fetchOmpCatalog(refresh = false): Promise<OmpCatalog> {
  return request<OmpCatalog>(`/models/omp${refresh ? "?refresh=true" : ""}`)
}

export async function saveModelDefaults(defaults: ModelDefaults): Promise<ModelsResponse> {
  return put<ModelsResponse>("/models/defaults", defaults)
}

export async function fetchBackendConfig(): Promise<BackendConfig> {
  return request<BackendConfig>("/config")
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
  return put(`/prompts/${encodeURIComponent(slug)}`, { content })
}

export async function deletePrompt(slug: string): Promise<void> {
  await del(`/prompts/${encodeURIComponent(slug)}`)
}

export async function savePromptOrder(order: string[]): Promise<void> {
  await put("/prompt-order", { order })
}

/** Tools and tool options are sent only when at least one tool is offered. */
export function createChatStream(req: ChatRequest): SSEStreamResult {
  const body: Record<string, unknown> = {
    model: req.model,
    chat_id: req.chat_id,
    user_msg: req.user_msg,
    system_prompt: req.system_prompt ?? "",
    downscale_images: req.downscale_images,
    img_paths: req.img_paths ?? [],
    messages: req.messages ?? [],
    project_slug: req.project_slug ?? null,
  }

  if (req.tools?.length) {
    body.tools = req.tools
    body.tool_options = req.tool_options ?? {}
  }
  if (req.reasoning_effort) body.reasoning_effort = req.reasoning_effort

  return createSSEStream("/chat", body)
}

export async function listChatHistories(): Promise<ChatHistoriesResponse> {
  return request<ChatHistoriesResponse>("/chat-histories")
}

export async function fetchBranchMeta(dirs?: string[]): Promise<Record<string, BranchMeta>> {
  const params = dirs && dirs.length > 0 ? `?dirs=${dirs.map(encodeURIComponent).join(",")}` : ""
  return request(`/chat-histories/branch-meta${params}`)
}

export async function loadChatHistory(filename: string): Promise<{ messages: Message[]; filename: string; chat_id: string | null; title: string | null; usage: Usage | null; parent_id: string | null; branch_message_idx: number | null; children: BranchMeta["children"] }> {
  return request(`/chat-histories/${encodePath(filename)}`)
}

/** Omitting a field leaves it unchanged server-side; passing null clears it. */
interface SaveChatHistoryOptions {
  chatId?: string
  title?: string
  usage?: Usage
  parentId?: string | null
  branchMessageIdx?: number | null
  children?: BranchMeta["children"] | null
}

export async function saveChatHistory(filename: string, messages: Message[] = [], opts: SaveChatHistoryOptions = {}): Promise<{ status: string; filename: string }> {
  const body: Record<string, unknown> = { messages, chat_id: opts.chatId ?? null, title: opts.title ?? null, usage: opts.usage ?? null }
  if (opts.parentId !== undefined) body.parent_id = opts.parentId
  if (opts.branchMessageIdx !== undefined) body.branch_message_idx = opts.branchMessageIdx
  if (opts.children !== undefined) body.children = opts.children
  return put(`/chat-histories/${filename}`, body)
}

export async function moveHistoryItem(filename: string, targetDir: string): Promise<{ status: string; new_path: string }> {
  return post("/chat-histories/move", { filename, target_dir: targetDir })
}

export async function createBranch(parentFile: string, branchMessageIdx: number): Promise<{ status: string; child_file: string; chat_id: string }> {
  return post("/chat-histories/branch", { parent_file: parentFile, branch_message_idx: branchMessageIdx })
}

export async function mergeBranch(childFile: string): Promise<{ status: string }> {
  return post("/chat-histories/merge", { child_file: childFile })
}

export async function cascadeDelete(filename: string): Promise<{ status: string; deleted: string[] }> {
  return del(`/chat-histories/cascade/${encodePath(filename)}`)
}

export function createOCRStream(
  imgBase64: string,
  model?: string
): SSEStreamResult {
  return createSSEStream("/ocr", { img_base64: imgBase64, model })
}

type AttachmentPayload = Omit<Attachment, "active" | "url">

function toChatAttachment(data: AttachmentPayload, chatId: string, slug: string | null): Attachment {
  return { ...data, active: true, url: chatFileUrl(chatId, data.name, slug) }
}

export async function uploadFile(chatId: string, file: File, slug: string | null = null, overwrite = false): Promise<Attachment> {
  const query = toQuery({ chat_id: chatId, slug, overwrite })
  return toChatAttachment(await request<AttachmentPayload>(`/files/upload${query}`, { method: "POST", body: fileForm(file) }), chatId, slug)
}

interface ParsedAttachment {
  name: string
  parsedMd: string | null
}

export async function parseFiles(chatId: string, filenames: string[], slug: string | null = null): Promise<ParsedAttachment[]> {
  const query = toQuery({ chat_id: chatId, filenames, slug })
  return request(`/files/parse${query}`, { method: "POST" })
}

export async function deleteAttachment(chatId: string, filename: string, slug: string | null = null): Promise<void> {
  await del(`/files/chat/${chatId}/att/${encodeURIComponent(filename)}${toQuery({ slug })}`)
}

export async function listRootFiles(): Promise<{ enabled: boolean; files: RootFile[] }> {
  return request("/fileroots/files")
}

/** Every mount's tree, a mount listed with one of its `worktrees` shows that worktree instead. */
export async function fileRootTree(worktrees: string[] = []): Promise<{ enabled: boolean; tree: FileTreeNode[] }> {
  return request(`/fileroots/tree${toQuery({ worktree: worktrees })}`)
}

export interface Worktree {
  path: string
  branch: string | null
  is_main: boolean
  changed_files: number
  ahead: number
  behind: number
}

export interface WorktreeList {
  /** The mount holds a `.git` folder, only clones have worktrees. */
  is_clone: boolean
  default_base: string | null
  default_branch?: string | null
  branches: string[]
  worktrees: Worktree[]
}

export async function fetchWorktrees(root: string): Promise<WorktreeList> {
  return request(`/fileroots/worktrees${toQuery({ root })}`)
}

/** Pull `base`, then branch worktree `name` off it, returns the new worktree's path. */
export async function createWorktree(root: string, name: string, base: string): Promise<RootFile> {
  return post("/fileroots/worktrees", { root, name, base })
}

/** Open the inferred default branch without pulling or switching the main checkout. */
export async function openDefaultWorktree(root: string): Promise<RootFile> {
  return post("/fileroots/worktrees/default", { root })
}

/** Throws ApiError 409 when the worktree holds uncommitted changes and `force` is off. */
export async function removeWorktree(root: string, path: string, force = false): Promise<void> {
  await del(`/fileroots/worktrees${toQuery({ root, path, force })}`)
}

/** Throws ApiError 409 when the branch is not fully merged and `force` is off. */
export async function deleteWorktreeBranch(root: string, branch: string, force = false): Promise<void> {
  await del(`/fileroots/worktrees/branch${toQuery({ root, branch, force })}`)
}

/** Add file-root files (folders expand to their valid files) to a project's documents. */
export async function addRootFilesToProject(slug: string, paths: string[]): Promise<ProjectDocument[]> {
  const data = await post<{ documents: ProjectDocument[] }>("/fileroots/add-to-project", { slug, paths })
  return data.documents
}

export type DiffStatus = "modified" | "added" | "deleted" | "renamed" | "untracked" | "conflict"

export interface RootDiffFiles {
  is_git_repository: boolean
  files: { path: string; status: DiffStatus; stamp: string; added: number | null; removed: number | null }[]
}

/** Files of the git working tree at a mounted root that differ from HEAD. */
export async function fetchRootDiffFiles(root: string): Promise<RootDiffFiles> {
  return request(`/fileroots/diff/files?root=${encodeURIComponent(root)}`)
}

/** Unified patch of one changed file, `supported` is false for binary or oversized files. */
export async function fetchRootFileDiff(root: string, path: string): Promise<{ supported: boolean; status?: DiffStatus; patch?: string }> {
  return request(`/fileroots/diff?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`)
}

export async function fetchRootDiffContent(root: string, path: string): Promise<{ base_content: string; content: string }> {
  return request(`/fileroots/diff/content?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`)
}

export async function addFileRoot(path: string): Promise<{ enabled: boolean; tree: FileTreeNode[] }> {
  return post("/fileroots/roots", { path })
}

export async function removeFileRoot(path: string): Promise<{ enabled: boolean; tree: FileTreeNode[] }> {
  return del(`/fileroots/roots?path=${encodeURIComponent(path)}`)
}

/** List a backend directory for picking a workspace; `~` is the backend user's home. */
export async function browseDirectories(path: string): Promise<DirectoryListing> {
  return request(`/fileroots/browse${toQuery({ path })}`)
}

export async function readFileRootRendered(path: string): Promise<string> {
  const data = await request<{ path: string; content: string }>(`/fileroots/rendered?path=${encodeURIComponent(path)}`)
  return data.content
}

export async function writeRootFile(path: string, content: string): Promise<void> {
  await post("/fileroots/file", { path, content })
}

/** Read a file's text content (markdown / unknown-treated-as-text). */
export async function loadFileViewerText(path: string): Promise<string> {
  const data = await request<{ path: string; content: string }>(`/fileviewer/text?path=${encodeURIComponent(path)}`)
  return data.content
}

/** Canvas snapshots carry their durable mutation revision for gapless stream resume. */
export async function loadCanvasSnapshot(canvasKey: string): Promise<CanvasSnapshot> {
  return request<CanvasSnapshot>(`/canvases/${encodePath(canvasKey)}`)
}

/** Resolves with the revision holding these mutations; the stream delivers their content. */
export async function submitCanvasMutations(canvasKey: string, mutations: CanvasMutation[]): Promise<{ revision: number }> {
  return post<{ revision: number }>(`/canvases/${encodePath(canvasKey)}/mutations`, { mutations })
}

/** Attach a FileRoots file as a live reference: no copy, content is read from the source. */
async function attachRootFile(path: string): Promise<Attachment> {
  const data = await request<{ name: string; mime: string; path: string; content?: string; parsedMd?: string }>(
    `/fileroots/attach?path=${encodeURIComponent(path)}`,
    { method: "POST" },
  )
  return { ...data, active: true, vaultPath: data.path, url: fileViewerRawUrl(data.path) }
}

/** MIME by extension for app documents, matching the backend's guess for the kinds the chat tells apart. */
function documentMime(path: string): string {
  const kind = kindOf(path)
  if (kind === "image") return `image/${extOf(path)}`
  if (kind === "pdf") return "application/pdf"
  return kind === "markdown" ? "text/markdown" : "text/plain"
}

/**
 * Attach any document as a live reference, re-read on every send.
 *
 * - Mount files (absolute paths) and library PDFs (`PDFs/<name>`) go through the file roots, which follow moved files
 * - App documents (logical keys) are read by key, text only
 */
export async function attachReference(path: string): Promise<Attachment> {
  if (path.startsWith("/") || path.startsWith("PDFs/")) return attachRootFile(path)
  const mime = documentMime(path)
  const content = mime.startsWith("text/") ? await loadFileViewerText(path) : undefined
  return { name: baseName(path), mime, url: fileViewerRawUrl(path), active: true, vaultPath: path, content }
}

/**
 * Save `content` to the source of the text reference `path` (an `Attachment.vaultPath`).
 *
 * - Mount files in place, project documents (`project/<slug>/document/<name>`) and loose notes (`note/<name>`) by name, architecture sources in their workspace
 * - Throws for sources that are not text documents (PDFs, images), so a text edit never overwrites them
 */
export async function writeReference(path: string, content: string): Promise<void> {
  if (!documentMime(path).startsWith("text/")) throw new Error(`Not a text document: ${path}`)
  if (path.startsWith("/")) return writeRootFile(path, content)
  if (isArchitecturePath(path)) {
    await writeArchitectureSource(architectureSlug(path), architectureSource(path), content)
    return
  }
  const projectDocument = /^project\/([^/]+)\/document\/([^/]+)$/.exec(path)
  if (projectDocument) {
    await writeDocument(projectDocument[1], projectDocument[2], content)
    return
  }
  const note = /^note\/([^/]+)$/.exec(path)
  if (note) {
    await writeDocument("", note[1], content)
    return
  }
  throw new Error(`Not a text document: ${path}`)
}

export async function listProjectDocuments(slug: string): Promise<ProjectDocument[]> {
  const data = await request<{ documents: ProjectDocument[] }>(`/documents?slug=${encodeURIComponent(slug)}`)
  return data.documents
}

export async function listAllDocuments(): Promise<ProjectDocument[]> {
  const data = await request<{ documents: ProjectDocument[] }>("/documents/all")
  return data.documents
}

export async function listNotes(): Promise<ProjectDocument[]> {
  const data = await request<{ documents: ProjectDocument[] }>("/documents/notes")
  return data.documents
}

export async function addDocument(slug: string, path: string): Promise<ProjectDocument> {
  return post<ProjectDocument>(`/documents/add?slug=${encodeURIComponent(slug)}`, { path })
}

export async function removeDocument(slug: string, path: string): Promise<void> {
  await del(`/documents${toQuery({ slug, path })}`)
}

export async function moveDocument(path: string, fromSlug: string, toSlug: string): Promise<ProjectDocument> {
  return post<ProjectDocument>("/documents/move", { path, from_slug: fromSlug, to_slug: toSlug })
}

export async function renameDocument(slug: string, path: string, name: string): Promise<ProjectDocument> {
  return post<ProjectDocument>("/documents/rename", { slug, path, name })
}

export async function writeDocument(slug: string, name: string, content: string = ""): Promise<ProjectDocument> {
  return post<ProjectDocument>("/documents/write", { slug, name, content })
}

// Architecture writes carry no revision. Structured operations are revalidated
// against the server's current model: one that no longer fits is rejected with a
// 400 and a reason. A source save replaces the file with the supplied text,
// whatever it held meanwhile, provided the model then parses without errors.
const architecturePath = (slug: string) => `/architecture/${encodeURIComponent(slug)}`

/** The whole model: every view, with saved positions applied; empty for a project without an architecture. */
export function readArchitecture(slug: string): Promise<ArchitectureWorkspace> {
  return request<ArchitectureWorkspace>(architecturePath(slug))
}

/** `file` is the editing window's source, where top-level elements drawn there are written; null for ops that name their own file, e.g. creating a system. */
export function applyArchitectureOperations(slug: string, ops: ArchitectureOperation[], file: string | null): Promise<ArchitectureWorkspace> {
  return post<ArchitectureWorkspace>(`${architecturePath(slug)}/ops`, { ops, file })
}

/** Replaces `path` with `content`; rejected unless the model then parses without errors. */
export function writeArchitectureSource(slug: string, path: string, content: string): Promise<ArchitectureWorkspace> {
  return put<ArchitectureWorkspace>(`${architecturePath(slug)}/source`, { path, content })
}

/** Adds a source file exactly as named, empty: for writing `.c4` by hand. */
export function createArchitectureSource(slug: string, path: string): Promise<ArchitectureWorkspace> {
  return post<ArchitectureWorkspace>(`${architecturePath(slug)}/source`, { path })
}

/** Deletes the files together, rejected while the remaining files still use what they declare. */
export function deleteArchitectureSources(slug: string, paths: string[]): Promise<ArchitectureWorkspace> {
  return del<ArchitectureWorkspace>(`${architecturePath(slug)}/source${toQuery({ path: paths })}`)
}

export async function writeBinaryDocument(slug: string, filename: string, blob: Blob): Promise<ProjectDocument> {
  return request<ProjectDocument>(`/documents/write-binary?slug=${encodeURIComponent(slug)}`, {
    method: "POST",
    body: fileForm(blob, filename),
  })
}

/** Store a rendered canvas (.jpg) as a database asset. */
export async function storeDrawing(filename: string, blob: Blob): Promise<void> {
  await request("/documents/store-drawing", { method: "POST", body: fileForm(blob, filename) })
}

export async function listProjects(): Promise<ProjectListItem[]> {
  const data = await request<{ projects: ProjectListItem[] }>("/projects")
  return data.projects
}

export async function loadProject(name: string): Promise<ProjectData> {
  return request<ProjectData>(`/projects/${encodeURIComponent(name)}`)
}

export async function createProject(name: string): Promise<ProjectData> {
  return post<ProjectData>("/projects", { name })
}

export async function deleteProject(name: string): Promise<void> {
  await del(`/projects/${encodeURIComponent(name)}`)
}

export async function updateProjectKanban(name: string, data: ProjectStateUpdate): Promise<ProjectData> {
  return put<ProjectData>(`/projects/${encodeURIComponent(name)}/state`, data)
}

export async function addProjectCard(name: string, title: string, description: string = "", state: string = "backlog"): Promise<KanbanCard> {
  return post<KanbanCard>(`/projects/${encodeURIComponent(name)}/cards`, { title, description, state })
}

export async function moveProjectCard(name: string, cardId: string, state: string): Promise<KanbanCard> {
  return patch<KanbanCard>(`/projects/${encodeURIComponent(name)}/cards/${cardId}`, { state })
}

export async function deleteProjectCard(name: string, cardId: string): Promise<void> {
  await del(`/projects/${encodeURIComponent(name)}/cards/${cardId}`)
}

/** Omitting a field leaves it unchanged server-side; passing null clears it. */
interface SaveProjectTabOptions {
  chatId?: string
  tabName?: string
  title?: string
  usage?: Usage
}

export async function saveProjectTab(name: string, filename: string, messages: Message[], opts: SaveProjectTabOptions = {}): Promise<{ status: string }> {
  const body: Record<string, unknown> = { filename, messages, chat_id: opts.chatId ?? null, tab_name: opts.tabName ?? null, title: opts.title ?? null, usage: opts.usage ?? null }
  return put(`/projects/${encodeURIComponent(name)}/tabs/${encodeURIComponent(filename)}`, body)
}

export async function extractMemories(
  messages: { role: string; content: string }[],
  projectSlug?: string | null,
  chatId?: string | null,
  scope?: "global" | "project" | null,
): Promise<MemoryExtractResponse> {
  return post<MemoryExtractResponse>("/memory/extract", {
    messages,
    project_slug: projectSlug ?? null,
    chat_id: chatId ?? null,
    scope: scope ?? null,
  })
}

export async function cancelMemories(reviewId: string, projectSlug?: string | null): Promise<void> {
  await post("/memory/cancel", { review_id: reviewId, project_slug: projectSlug ?? null })
}

export async function commitMemoryDoc(
  scope: string,
  acceptedMemories: ProposedMemory[],
  projectSlug: string | null,
  reviewId: string | null,
  revisedMemories?: PreviewMemory[] | null,
): Promise<void> {
  await post("/memory/commit", {
    scope,
    accepted_memories: acceptedMemories,
    project_slug: projectSlug,
    review_id: reviewId,
    ...(revisedMemories ? { revised_memories: revisedMemories } : {}),
  })
}

export async function previewMemoryDoc(
  scope: string,
  acceptedMemories: ProposedMemory[],
  projectSlug: string | null,
): Promise<MemoryPreviewResponse> {
  return post("/memory/preview", {
    scope,
    accepted_memories: acceptedMemories,
    project_slug: projectSlug,
  })
}

export async function getMemories(scope: "global" | "project", projectSlug?: string | null): Promise<{ memories: PreviewMemory[] }> {
  return request(`/memory/memories${toQuery({ scope, project_slug: projectSlug })}`)
}

export async function getCategories(scope: "global" | "project", projectSlug?: string | null): Promise<{ categories: CategoryDef[] }> {
  return request(`/memory/categories${toQuery({ scope, project_slug: projectSlug })}`)
}

export async function saveCategories(scope: "global" | "project", categories: CategoryDef[], projectSlug?: string | null): Promise<void> {
  await put("/memory/categories", { scope, categories, project_slug: projectSlug ?? null })
}

export async function remapOrphanedCategory(
  orphanedMemories: PreviewMemory[],
  remainingCategories: CategoryDef[],
  projectSlug?: string | null,
): Promise<{ memories: PreviewMemory[] }> {
  return post("/memory/remap-category", {
    orphaned_memories: orphanedMemories,
    remaining_categories: remainingCategories,
    project_slug: projectSlug ?? null,
  })
}

export async function fetchOmpOverview(): Promise<OmpOverview> {
  return request<OmpOverview>("/omp/overview")
}

export async function addOmpWorkspace(path: string): Promise<OmpWorkspace> {
  return post<OmpWorkspace>("/omp/workspaces", { path })
}

export async function fetchOmpArchive(): Promise<OmpArchivedSession[]> {
  return request<OmpArchivedSession[]>("/omp/archive")
}

/** Restore an archived omp session, resolves to its session id. */
export async function restoreOmpSession(key: string): Promise<string> {
  const data = await post<{ session_id: string }>("/omp/archive/restore", { key })
  return data.session_id
}

/** Archive an omp session. omp-web refuses (409) while forks of it exist. */
export async function archiveOmpSession(sessionId: string): Promise<void> {
  await ensureOk(await fetch(`${getApiBase()}/omp/sessions/${encodeURIComponent(sessionId)}/archive`, { method: "POST", headers: deviceHeaders() }))
}

export async function fetchOmpSystemPrompt(sessionId: string): Promise<OmpSystemPrompt> {
  return request<OmpSystemPrompt>(`/omp/sessions/${encodeURIComponent(sessionId)}/system-prompt`)
}

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
  return request<FactoryWorkflow[]>(`/factory/workflows?repo=${encodeURIComponent(repo)}`)
}

export function fetchFactoryTasks(repo: string): Promise<FactoryTask[]> {
  return request<FactoryTask[]>(`/factory/tasks?repo=${encodeURIComponent(repo)}`)
}

export function fetchFactoryTask(id: string): Promise<FactoryTaskDetail> {
  return request<FactoryTaskDetail>(`/factory/tasks/${encodeURIComponent(id)}`)
}

export function fetchFactoryEvents(repo: string, after: number): Promise<FactoryEvent[]> {
  return request<FactoryEvent[]>(`/factory/events?repo=${encodeURIComponent(repo)}&after=${after}`)
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
