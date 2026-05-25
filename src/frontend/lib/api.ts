import type { ChatHistoriesResponse, ChatRequest, Message, ModelsResponse, ResearchRequest, TavilySearchRequest, TavilySearchResult } from "./types"

const BASE = "/api"

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options)
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

export function createChatStream(
  req: ChatRequest
): { stream: ReadableStreamDefaultReader<Uint8Array>; abort: () => void } {
  const controller = new AbortController()

  const body: Record<string, unknown> = {
    model: req.model,
    user_msg: req.user_msg,
    system_prompt: req.system_prompt ?? "",
    temperature: req.temperature ?? 0.2,
    top_p: req.top_p ?? 0.95,
    downscale_images: req.downscale_images ?? true,
  }
  if (req.reasoning_effort) body.reasoning_effort = req.reasoning_effort
  if (req.img_base64) body.img_base64 = req.img_base64

  const promise = fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: controller.signal,
  })

  return {
    stream: new ReadableStream({
      async start(controller_stream) {
        const res = await promise
        if (!res.ok || !res.body) {
          controller_stream.error(await res.text())
          return
        }
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        let eventData: string[] = []

        async function pump(): Promise<void> {
          const { done, value } = await reader.read()
          if (done) {
            if (eventData.length > 0) {
              controller_stream.enqueue(new TextEncoder().encode(eventData.join("\n")))
            }
            controller_stream.close()
            return
          }
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n")
          buffer = lines.pop() ?? ""

          for (let line of lines) {
            if (line.endsWith("\r")) line = line.slice(0, -1)
            
            if (!line) {
              if (eventData.length > 0) {
                controller_stream.enqueue(new TextEncoder().encode(eventData.join("\n")))
                eventData = []
              }
              continue
            }
            if (line.startsWith("event:")) continue
            if (line.startsWith("data:")) {
              let d = line.startsWith("data: ") ? line.slice(6) : line.slice(5)
              eventData.push(d)
            }
          }
          return pump()
        }
        return pump()
      },
    }).getReader(),
    abort: () => controller.abort(),
  }
}

export async function listChatHistories(): Promise<ChatHistoriesResponse> {
  return request<ChatHistoriesResponse>("/chat-histories")
}

export async function loadChatHistory(filename: string): Promise<{ messages: Message[]; filename: string }> {
  return request(`/chat-histories/${filename}`)
}

export async function saveChatHistory(filename: string): Promise<{ status: string; filename: string }> {
  return request(`/chat-histories/${filename}`, { method: "PUT" })
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

export async function runTavilySearch(req: TavilySearchRequest): Promise<TavilySearchResult> {
  return request<TavilySearchResult>("/tavily-search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  })
}

export async function downscaleImage(imgBase64: string, maxTokens: number = 4096): Promise<string> {
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
): { stream: ReadableStreamDefaultReader<Uint8Array>; abort: () => void } {
  const controller = new AbortController()
  const promise = fetch(`${BASE}/ocr`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ img_base64: imgBase64, model }),
    signal: controller.signal,
  })
  return {
    stream: new ReadableStream({
      async start(ctrl) {
        const res = await promise
        if (!res.ok || !res.body) { ctrl.error(await res.text()); return }
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        let eventData: string[] = []
        async function pump(): Promise<void> {
          const { done, value } = await reader.read()
          if (done) { 
            if (eventData.length > 0) {
              ctrl.enqueue(new TextEncoder().encode(eventData.join("\n")))
            }
            ctrl.close(); 
            return 
          }
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n")
          buffer = lines.pop() ?? ""
          for (let line of lines) {
            if (line.endsWith("\r")) line = line.slice(0, -1)
            
            if (!line) {
              if (eventData.length > 0) {
                ctrl.enqueue(new TextEncoder().encode(eventData.join("\n")))
                eventData = []
              }
              continue
            }
            if (line.startsWith("event:")) continue
            if (line.startsWith("data:")) {
              let d = line.startsWith("data: ") ? line.slice(6) : line.slice(5)
              eventData.push(d)
            }
          }
          return pump()
        }
        return pump()
      },
    }).getReader(),
    abort: () => controller.abort(),
  }
}
