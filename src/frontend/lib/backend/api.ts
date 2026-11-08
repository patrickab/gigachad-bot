import type { Attachment, BackendConfig, ChatRequest, LearnSession, Message, ModelDefaults, ModelProvider, ModelsResponse, OmpArchivedSession, OmpCatalog, OmpOverview, OmpSystemPrompt, PromptMeta, ProviderUsageReport, ReasoningLevels, Usage } from "@/lib/domain/types"
import type { DiffStatus, DirectoryListing, FileTreeNode, RootDiffFiles, RootFile, WorktreeList } from "@/lib/domain/files"
import type { BranchMeta, ChatHistoriesResponse, ChatHistory } from "@/lib/domain/histories"
import type { KanbanCard, ProjectData, ProjectDocument, ProjectListItem } from "@/lib/domain/projects"
import type { MemoryExtractResponse, MemoryOperation, MemoryScope, MemoryTopic, StoredMemory } from "@/lib/domain/memory"
import type { FactoryDefaults, FactoryEvent, FactoryMessage, FactoryPreview, FactoryRepo, FactoryTask, FactoryTaskDetail, FactoryWorkflow, TaskAction, TaskCreate } from "@/lib/domain/factory"
import { createSSEStream } from "./sse"
import type { SSEStreamResult } from "./sse"
import { getApiBase } from "./config"
import type { ArchitectureOperation, ArchitectureWorkspace } from "@/lib/domain/architecture"
import type { CanvasMutation, CanvasSnapshot } from "@/lib/domain/canvasDocument"
import { deviceHeaders } from "./deviceId"
import type { KernelSpecs, RunningKernel } from "@/lib/domain/jupyter"
import { chatFileUrl, encodePath, fileViewerRawUrl, toQuery } from "./urls"
import type { LatexBuild, LatexTemplates, StoredImage, SynctexClickedSource, SynctexSource, SynctexView, TemplateSetInfo, TexProject } from "@/lib/domain/latex"
import { parseHistoryFile } from "@/lib/domain/historyFile"

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

// Chrome pools uncredentialed and credentialed requests on separate connections. Tailscale Serve can drop an
// HTTP/2 connection without a reset, and the browser reuses it until TCP gives up, so a stuck caller switches pools.
let connectionPool = 0

export function currentConnectionPool(): number {
  return connectionPool
}

/** Send later requests over a fresh connection, once per stuck pool however many callers report it. */
export function leaveConnectionPool(pool: number): void {
  if (pool === connectionPool) connectionPool += 1
}

/** The single place that adds the API base URL and the device headers, the status is left to the caller. */
function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = { ...(init?.headers as Record<string, string> | undefined), ...deviceHeaders() }
  const credentials: RequestCredentials = connectionPool % 2 ? "include" : "same-origin"
  return fetch(`${getApiBase()}${path}`, { credentials, ...init, headers })
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await ensureOk(await apiFetch(path, options))
  return res.json()
}

/** Like `request` for endpoints whose response body is not read. */
async function requestVoid(path: string, init?: RequestInit): Promise<void> {
  await ensureOk(await apiFetch(path, init))
}

function jsonInit(method: string, body: unknown, headers: Record<string, string> = {}): RequestInit {
  return { method, headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) }
}

/** POSTs `body` as JSON and streams the SSE response. */
function sseStream(path: string, body: Record<string, unknown>): SSEStreamResult {
  return createSSEStream((signal) => apiFetch(path, { ...jsonInit("POST", body), signal }))
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
  // keepalive: a deletion committed while the page unloads still reaches the backend
  return request<T>(path, { method: "DELETE", keepalive: true })
}

function fileForm(file: File | Blob, filename?: string): FormData {
  const form = new FormData()
  if (filename !== undefined) form.append("file", file, filename)
  else form.append("file", file)
  return form
}

export async function fetchModels(signal?: AbortSignal): Promise<ModelsResponse> {
  return request<ModelsResponse>("/models", { signal })
}

/** Subscription usage per OMP login; `refresh` skips the server's 5 minute cache. */
export async function fetchProviderUsage(refresh = false): Promise<{ reports: ProviderUsageReport[] }> {
  return request<{ reports: ProviderUsageReport[] }>(`/provider-usage${toQuery({ refresh })}`)
}

export async function fetchReasoningLevels(model: string, signal?: AbortSignal): Promise<ReasoningLevels> {
  return request<ReasoningLevels>(`/models/reasoning-levels${toQuery({ model })}`, { signal })
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

export async function fetchBackendConfig(signal?: AbortSignal): Promise<BackendConfig> {
  return request<BackendConfig>("/config", { signal })
}

export async function fetchPrompts(signal?: AbortSignal): Promise<Record<string, string>> {
  const data = await request<{ prompts: Record<string, string> }>("/prompts", { signal })
  return data.prompts
}

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

  return sseStream("/chat", body)
}

export async function listChatHistories(signal?: AbortSignal): Promise<ChatHistoriesResponse> {
  return request<ChatHistoriesResponse>("/chat-histories", { signal })
}

export async function fetchBranchMeta(dirs?: string[], signal?: AbortSignal): Promise<Record<string, BranchMeta>> {
  const params = dirs && dirs.length > 0 ? `?dirs=${dirs.map(encodeURIComponent).join(",")}` : ""
  return request(`/chat-histories/branch-meta${params}`, { signal })
}

export async function loadChatHistory(filename: string): Promise<ChatHistory> {
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
  return sseStream("/ocr", { img_base64: imgBase64, model })
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

export async function listRootFiles(signal?: AbortSignal): Promise<{ enabled: boolean; files: RootFile[] }> {
  return request("/fileroots/files", { signal })
}

/** Whether any file root is mounted, without listing its files. */
export async function fileRootsEnabled(signal?: AbortSignal): Promise<{ enabled: boolean }> {
  return request("/fileroots/enabled", { signal })
}

type RootTree = { enabled: boolean; tree: FileTreeNode[] }

/** Every mount's tree, a mount listed with one of its `worktrees` shows that worktree instead. */
export async function fileRootTree(worktrees: string[] = [], signal?: AbortSignal): Promise<RootTree> {
  return request(`/fileroots/tree${toQuery({ worktree: worktrees })}`, { signal })
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

type DocumentList = { documents: ProjectDocument[] }
const documentsOf = (data: DocumentList) => data.documents

/** Add file-root files (folders expand to their valid files) to a project's documents. */
export async function addRootFilesToProject(slug: string, paths: string[]): Promise<ProjectDocument[]> {
  return post<DocumentList>("/fileroots/add-to-project", { slug, paths }).then(documentsOf)
}

/** Files of the git working tree at a mounted root that differ from HEAD. */
export async function fetchRootDiffFiles(root: string): Promise<RootDiffFiles> {
  return request(`/fileroots/diff/files${toQuery({ root })}`)
}

/** Unified patch of one changed file, `supported` is false for binary or oversized files. */
export async function fetchRootFileDiff(root: string, path: string): Promise<{ supported: boolean; status?: DiffStatus; patch?: string }> {
  return request(`/fileroots/diff${toQuery({ root, path })}`)
}

export async function fetchRootDiffContent(root: string, path: string): Promise<{ base_content: string; content: string }> {
  return request(`/fileroots/diff/content${toQuery({ root, path })}`)
}

export async function addFileRoot(path: string): Promise<RootTree> {
  return post("/fileroots/roots", { path })
}

/** Clone an https or ssh git link into `~/git/<repository>` on the backend host and mount it. */
export async function cloneFileRoot(url: string): Promise<RootTree> {
  return post("/fileroots/clone", { url })
}

export async function removeFileRoot(path: string): Promise<RootTree> {
  return del(`/fileroots/roots${toQuery({ path })}`)
}

/** List a backend directory for picking a workspace; `~` is the backend user's home. */
export async function browseDirectories(path: string): Promise<DirectoryListing> {
  return request(`/fileroots/browse${toQuery({ path })}`)
}

export async function readFileRootRendered(path: string): Promise<string> {
  const data = await request<{ path: string; content: string }>(`/fileroots/rendered${toQuery({ path })}`)
  return data.content
}

export async function writeRootFile(path: string, content: string): Promise<void> {
  await post("/fileroots/file", { path, content })
}

/** Read a file's text content (markdown / unknown-treated-as-text). */
export async function loadFileViewerText(path: string): Promise<string> {
  const data = await request<{ path: string; content: string }>(`/fileviewer/text${toQuery({ path })}`)
  return data.content
}

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

/** Attach a FileRoots file as a live reference: no copy, content is read from the source. */
export async function attachRootFile(path: string): Promise<Attachment> {
  const data = await request<{ name: string; mime: string; path: string; content?: string; parsedMd?: string }>(
    `/fileroots/attach${toQuery({ path })}`,
    { method: "POST" },
  )
  return { ...data, active: true, vaultPath: data.path, url: fileViewerRawUrl(data.path) }
}

export async function listProjectDocuments(slug: string, signal?: AbortSignal): Promise<ProjectDocument[]> {
  return request<DocumentList>(`/documents${toQuery({ slug })}`, { signal }).then(documentsOf)
}

export async function listAllDocuments(): Promise<ProjectDocument[]> {
  return request<DocumentList>("/documents/all").then(documentsOf)
}

export async function listNotes(signal?: AbortSignal): Promise<ProjectDocument[]> {
  return request<DocumentList>("/documents/notes", { signal }).then(documentsOf)
}

export async function addDocument(slug: string, path: string): Promise<ProjectDocument> {
  return post<ProjectDocument>(`/documents/add${toQuery({ slug })}`, { path })
}

/** Delete `path` from project `slug`; resolves with the keys and absolute paths that are gone (a mounted file only unlinked stays out). */
export async function removeDocument(slug: string, path: string): Promise<string[]> {
  return (await del<{ removed: string[] }>(`/documents${toQuery({ slug, path })}`)).removed
}

export async function moveDocument(path: string, fromSlug: string, toSlug: string): Promise<ProjectDocument> {
  return post<ProjectDocument>("/documents/move", { path, from_slug: fromSlug, to_slug: toSlug })
}

/** Move the stored project file `path` into the file-root `folder`, every project then references the file on disk. */
export async function moveDocumentToFolder(path: string, folder: string): Promise<ProjectDocument> {
  return post<ProjectDocument>("/documents/move-to-folder", { path, folder })
}

/** The renamed document, and the absolute paths of other files the rename rewrote to follow it (references to it). */
export type RenamedDocument = ProjectDocument & { changed: string[] }

export async function renameDocument(slug: string, path: string, name: string): Promise<RenamedDocument> {
  return post<RenamedDocument>("/documents/rename", { slug, path, name })
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
  return request<ProjectDocument>(`/documents/write-binary${toQuery({ slug })}`, {
    method: "POST",
    body: fileForm(blob, filename),
  })
}

/** Store a rendered canvas (.jpg) as a database asset. */
export async function storeDrawing(filename: string, blob: Blob): Promise<void> {
  await request("/documents/store-drawing", { method: "POST", body: fileForm(blob, filename) })
}

export async function listProjects(signal?: AbortSignal): Promise<ProjectListItem[]> {
  const data = await request<{ projects: ProjectListItem[] }>("/projects", { signal })
  return data.projects
}

const projectPath = (slug: string) => `/projects/${encodeURIComponent(slug)}`

export async function loadProject(name: string): Promise<ProjectData> {
  return request<ProjectData>(projectPath(name))
}

export async function createProject(name: string): Promise<ProjectData> {
  return post<ProjectData>("/projects", { name })
}

export async function deleteProject(name: string): Promise<void> {
  await del(projectPath(name))
}

export async function setProjectArchived(slug: string, archived: boolean): Promise<void> {
  await patch(projectPath(slug), { archived })
}

export function renameProject(slug: string, name: string): Promise<{ name: string; slug: string; archived: boolean }> {
  return patch(projectPath(slug), { name })
}

export async function addProjectCard(name: string, title: string, description: string = "", state: string = "backlog"): Promise<KanbanCard> {
  return post<KanbanCard>(`${projectPath(name)}/cards`, { title, description, state })
}

export async function moveProjectCard(name: string, cardId: string, state: string): Promise<KanbanCard> {
  return patch<KanbanCard>(`${projectPath(name)}/cards/${cardId}`, { state })
}

export async function deleteProjectCard(name: string, cardId: string): Promise<void> {
  await del(`${projectPath(name)}/cards/${cardId}`)
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
  return put(`${projectPath(name)}/tabs/${encodeURIComponent(filename)}`, body)
}

/** Save a chat to `historyFile`: through its project's tab when it belongs to `projectSlug`, as a plain history otherwise. */
export async function saveChat(historyFile: string, projectSlug: string | null, messages: Message[], opts: { chatId: string; title: string; tabName?: string; usage?: Usage }): Promise<void> {
  const { slug, filename } = parseHistoryFile(historyFile)
  const { tabName, ...historyOpts } = opts
  if (slug && slug === projectSlug) await saveProjectTab(slug, filename, messages, { ...historyOpts, tabName })
  else await saveChatHistory(historyFile, messages, historyOpts)
}

/** Propose memory operations for the global scope and, with `projectSlug`, the project scope. */
export async function extractMemories(
  messages: { role: string; content: string }[],
  projectSlug?: string | null,
  chatId?: string | null,
): Promise<MemoryExtractResponse> {
  return post<MemoryExtractResponse>("/memory/extract", {
    messages,
    project_slug: projectSlug ?? null,
    chat_id: chatId ?? null,
  })
}

export async function cancelMemories(reviewId: string): Promise<void> {
  await post("/memory/cancel", { review_id: reviewId })
}

/** Apply accepted operations and topics of one scope, advancing the review's watermark. */
export async function commitMemories(
  scope: MemoryScope,
  operations: MemoryOperation[],
  topics: MemoryTopic[],
  projectSlug: string | null,
  reviewId: string,
): Promise<void> {
  await post("/memory/commit", { scope, operations, topics, project_slug: projectSlug, review_id: reviewId })
}

export async function getMemories(scope: MemoryScope, projectSlug?: string | null): Promise<{ memories: StoredMemory[] }> {
  return request(`/memory/memories${toQuery({ scope, project_slug: projectSlug })}`)
}

export async function saveMemories(scope: MemoryScope, memories: StoredMemory[], projectSlug?: string | null): Promise<void> {
  await put("/memory/memories", { scope, memories, project_slug: projectSlug ?? null })
}

export async function getTopics(scope: MemoryScope, projectSlug?: string | null, signal?: AbortSignal): Promise<{ topics: MemoryTopic[] }> {
  return request(`/memory/topics${toQuery({ scope, project_slug: projectSlug })}`, { signal })
}

export async function saveTopics(scope: MemoryScope, topics: MemoryTopic[], projectSlug?: string | null): Promise<void> {
  await put("/memory/topics", { scope, topics, project_slug: projectSlug ?? null })
}

/** Proposed regrouping of a scope's facts into subject topics, nothing is written. */
export async function regroupMemories(scope: MemoryScope, projectSlug?: string | null): Promise<{ topics: MemoryTopic[]; memories: StoredMemory[] }> {
  return post("/memory/regroup", { scope, project_slug: projectSlug ?? null })
}

export async function fetchOmpOverview(signal?: AbortSignal): Promise<OmpOverview> {
  return request<OmpOverview>("/omp/overview", { signal })
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
  await requestVoid(`/omp/sessions/${encodeURIComponent(sessionId)}/archive`, { method: "POST" })
}

/** Delete an omp session with its file and artifacts. Its forks are re-parented, not deleted. */
export async function deleteOmpSession(sessionId: string): Promise<void> {
  await requestVoid(`/omp/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" })
}

export async function fetchOmpSystemPrompt(sessionId: string): Promise<OmpSystemPrompt> {
  return request<OmpSystemPrompt>(`/omp/sessions/${encodeURIComponent(sessionId)}/system-prompt`)
}

/** An omp session's active branch as a chat history: user and assistant text, no thinking or tool calls. */
export async function fetchOmpSessionHistory(sessionId: string): Promise<Pick<Message, "role" | "content">[]> {
  return request<Pick<Message, "role" | "content">[]>(`/omp/sessions/${encodeURIComponent(sessionId)}/history`)
}

/** What `prepareWorkbench` resolves: the file's absolute disk path and the directory its omp sessions run in. The response's `kind` is not read, the client derives it from the path. */
export interface PreparedWorkbench {
  path: string
  cwd: string
}

/** Get the workbench file *path* ready: a stored document moves to disk first, its new path comes back. */
export function prepareWorkbench(path: string): Promise<PreparedWorkbench> {
  return post<PreparedWorkbench>("/workbench/prepare", { path })
}

/**
 * Create workbench document *name* (`.tex` or `.md`) of project *slug* on disk, listed in its files. Resolves to its path.
 *
 * - *template* names a LaTeX template set: the document becomes a presentation folder `<stem>/main.tex` with the theme beside it
 */
export async function createWorkbenchDocument(slug: string, name: string, template?: string): Promise<string> {
  return (await post<{ path: string }>("/workbench/create", { slug, name, ...(template ? { template } : {}) })).path
}

/**
 * The omp sessions of one file's chat: `active` is null while a new session starts, `sessionIds` lists every session the file had, first linked first.
 *
 * - `moved` are sessions the file moved away from (a rename), each has to be told its new path once
 */
export interface FileSessions {
  active: string | null
  sessionIds: string[]
  moved: string[]
}

interface LinkedSessions {
  session_id: string | null
  session_ids: string[]
  moved: string[]
}

function toFileSessions(linked: LinkedSessions): FileSessions {
  return { active: linked.session_id, sessionIds: linked.session_ids, moved: linked.moved }
}

/** The omp sessions linked to the file at absolute *path*. */
export async function loadFileSessions(path: string): Promise<FileSessions> {
  return toFileSessions(await request<LinkedSessions>(`/workbench/session${toQuery({ path })}`))
}

/** Make *sessionId* the active session of the file at absolute *path*, null starts a new one. Resolves to the file's sessions. */
export async function saveFileSession(path: string, sessionId: string | null): Promise<FileSessions> {
  return toFileSessions(await put<LinkedSessions>("/workbench/session", { path, session_id: sessionId }))
}

/** *sessionId* was told where the file at absolute *path* is now. Resolves to the file's sessions. */
export async function informFileSession(path: string, sessionId: string): Promise<FileSessions> {
  return toFileSessions(await post<LinkedSessions>("/workbench/session/informed", { path, session_id: sessionId }))
}

/** Compile the `.tex` file at *path*. A TeX error resolves with `ok: false`, a missing TeX install rejects with ApiError 503. */
export function buildLatex(path: string): Promise<LatexBuild> {
  return post<LatexBuild>("/latex/build", { path })
}

/** The LaTeX project of the file at absolute *path*: its folder, root `.tex` and, when no root was found, the candidates. */
export function loadTexProject(path: string): Promise<TexProject> {
  return request<TexProject>(`/latex/project${toQuery({ path })}`)
}

/** The source line and column SyncTeX finds for the spot `x`, `y` (PDF points from the top-left corner) on 1-based *page* of the PDF the root at absolute *root* built. The double-clicked *word* picks its own line of a beamer frame, whose whole body SyncTeX puts on `\end{frame}`, then the clicked character's 0-based *offset* in it and the word's 0-based *occurrence* among equal words of its rendered line give the column. Rejects with ApiError 404 when there is none, 503 without `synctex`. */
export function findSynctexSource(root: string, page: number, x: number, y: number, word = "", offset = 0, occurrence = 0): Promise<SynctexClickedSource> {
  return post<SynctexClickedSource>("/latex/synctex", { path: root, page, x, y, ...(word ? { word, offset, occurrence } : {}) })
}

/** The PDF spot SyncTeX finds for 1-based *line* of the source *file* (absolute) in the PDF the root at absolute *root* built. Rejects with ApiError 404 when there is none, 403 for a file outside the roots, 503 without `synctex`. */
export function findSynctexView(root: string, file: string, line: number): Promise<SynctexView> {
  return post<SynctexView>("/latex/synctex/view", { path: root, file, line })
}

/** The source line after the slide on 1-based *page* of the PDF the root at absolute *root* built, where Neovim's cursor makes "Add slide" put the next frame: the slide's `\end{frame}`, or the heading of a theme's chapter slide. Rejects with ApiError 404 when there is none, 403 for a file outside the roots. */
export function findSlideEnd(root: string, page: number): Promise<SynctexSource> {
  return post<SynctexSource>("/latex/synctex/slide", { path: root, page })
}

/** Make *root* the root of the `.tex` at *path* by writing its `% !TEX root` line. Resolves to the project. */
export function chooseTexRoot(path: string, root: string): Promise<TexProject> {
  return post<TexProject>("/latex/root", { path, root })
}

/** Create file *name* in LaTeX project *folder*, a `.tex` starts with its `% !TEX root` line. Resolves to its path. */
export async function createTexProjectFile(folder: string, name: string): Promise<string> {
  return (await post<{ path: string }>("/latex/file", { folder, name })).path
}

/** Turn project *slug*'s workbench `.tex` at *path* into its LaTeX project folder entry. Resolves to the folder. */
export async function promoteToTexProject(slug: string, path: string): Promise<string> {
  return (await post<{ folder: string }>("/workbench/promote", { slug, path })).folder
}

/** Store a pasted PNG in the image folder next to the workbench file at absolute *path*, resolving to it, its placeholder path and the line to insert. */
export function storeWorkbenchImage(path: string, image: Blob): Promise<StoredImage> {
  return request<StoredImage>(`/workbench/image${toQuery({ path })}`, { method: "POST", body: fileForm(image, "image.png") })
}

/** The template set of the LaTeX root at absolute *root*, `theme` null when it is not a presentation. */
export function loadLatexTemplates(root: string): Promise<LatexTemplates> {
  return request<LatexTemplates>(`/latex/templates${toQuery({ root })}`)
}

/** The template sets a new presentation can start from. */
export function loadTemplateSets(): Promise<TemplateSetInfo[]> {
  return request<TemplateSetInfo[]>("/latex/template-sets")
}

/** Every learn session with the project it is grouped under. Host-access logins only. */
export async function fetchLearnSessions(signal?: AbortSignal): Promise<LearnSession[]> {
  return request<LearnSession[]>("/learn/sessions", { signal })
}

/** Group a learn session under `projectSlug` (out of any other project), or under none. */
export async function assignLearnSession(sessionId: string, projectSlug: string | null): Promise<void> {
  await requestVoid(`/learn/sessions/${encodeURIComponent(sessionId)}/project`, jsonInit("PUT", { project_slug: projectSlug }))
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

/** Start the production redeploy, the backend restarts shortly after. Host-access logins only. */
export async function startRedeploy(): Promise<void> {
  await requestVoid("/deploy/redeploy", { method: "POST" })
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

/** Where the notebook at absolute `path` works, its kernel and its chat: the mounted root holding it, else its folder. */
export async function fetchNotebookDirectory(path: string, signal?: AbortSignal): Promise<string> {
  return (await request<{ path: string }>(`/jupyter/directory${toQuery({ path })}`, { signal })).path
}

/** The kernel environments for the notebook at absolute `path`, and the one it opens with. */
export function fetchKernelSpecs(path: string, signal?: AbortSignal): Promise<KernelSpecs> {
  return request<KernelSpecs>(`/jupyter/kernelspecs${toQuery({ path })}`, { signal })
}

/** Start (or reuse, same name) the kernel of the notebook at `path`. Another name replaces the running one. */
export function startKernel(path: string, name: string): Promise<RunningKernel> {
  return post<RunningKernel>("/jupyter/kernel", { path, name })
}

/** Shut the kernel of the notebook at `path` down, whether or not one runs. */
export function shutdownKernel(path: string): Promise<void> {
  return requestVoid(`/jupyter/kernel${toQuery({ path })}`, { method: "DELETE" })
}
