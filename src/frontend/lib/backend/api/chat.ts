// Chat stream and chat history routes.
import type { SSEStreamResult } from "../sse"
import { encodePath } from "../urls"
import { del, post, put, request, sseStream } from "./http"
import type { BranchMeta, ChatHistoriesResponse, ChatHistory } from "@/lib/domain/histories"
import type { ChatRequest, Message, Usage } from "@/lib/domain/types"

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
