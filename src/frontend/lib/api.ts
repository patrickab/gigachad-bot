import type { ChatHistoriesResponse, ChatRequest, Message, MineruBatchResponse, MineruResult, ModelsResponse, ResearchRequest, ResearchTrace, ResearchTraceSummary } from "./types"
import { createSSEStream } from "./sse"
import type { SSEStreamResult } from "./sse"
import { API_BASE, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_DOWNSCALE_IMAGES, IMAGE_DOWNSCALE_MAX } from "./config"

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) throw new Error(await res.text())
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
    top_p: req.top_p ?? DEFAULT_TOP_P,
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

export async function loadChatHistory(filename: string): Promise<{ messages: Message[]; filename: string }> {
  return request(`/chat-histories/${filename}`)
}

export async function saveChatHistory(filename: string, messages: Message[] = []): Promise<{ status: string; filename: string }> {
  return request(`/chat-histories/${filename}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  })
}

export async function deleteChatHistory(filename: string): Promise<void> {
  await request(`/chat-histories/${filename}`, { method: "DELETE" })
}

export async function archiveChatHistory(filename: string): Promise<void> {
  await request(`/chat-histories/${filename}/archive`, { method: "PUT" })
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

export async function listResearchTraces(): Promise<{ traces: ResearchTraceSummary[] }> {
  return request("/research-traces")
}

export async function getResearchTrace(runId: string): Promise<ResearchTrace> {
  return request(`/research-traces/${runId}`)
}

export async function downscaleImage(imgBase64: string, maxTokens: number = IMAGE_DOWNSCALE_MAX): Promise<string> {
  const data = await request<{ img_base64: string }>("/downscale-image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ img_base64: imgBase64, max_tokens: maxTokens }),
  })
  return data.img_base64
}

export function createOCRStream(
  imgBase64: string,
  model: string
): SSEStreamResult {
  return createSSEStream("/ocr", { img_base64: imgBase64, model })
}

export async function parsePdf(file: File, query = "", backend = "pipeline", model = ""): Promise<MineruResult> {
  const form = new FormData()
  form.append("file", file)
  if (query) form.append("query", query)
  if (backend !== "pipeline") form.append("backend", backend)
  if (model) form.append("model", model)
  const res = await fetch(`${API_BASE}/mineru/parse`, { method: "POST", body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function parsePdfs(files: File[], query = "", backend = "pipeline", model = ""): Promise<MineruBatchResponse> {
  const form = new FormData()
  for (const file of files) form.append("files", file)
  if (query) form.append("query", query)
  if (backend !== "pipeline") form.append("backend", backend)
  if (model) form.append("model", model)
  const res = await fetch(`${API_BASE}/mineru/parse-batch`, { method: "POST", body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}