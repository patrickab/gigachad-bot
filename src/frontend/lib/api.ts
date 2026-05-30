import type { ChatHistoriesResponse, ChatRequest, Message, ModelsResponse, ResearchRequest } from "./types"
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

export async function runResearch(req: ResearchRequest): Promise<{ report: string; sources: string[]; costs: number }> {
  return request("/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  })
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