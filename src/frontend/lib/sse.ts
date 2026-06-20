import { API_BASE } from "./config"

export interface SSEEvent {
  event: string
  data: string
}

export interface SSEStreamResult {
  abort: () => void
  [Symbol.asyncIterator]: () => AsyncIterator<SSEEvent>
}

export async function* readLines(res: Response): AsyncGenerator<string> {
  if (!res.body) throw new Error("No response body")
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        if (buffer) yield buffer
        break
      }
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split("\n")
      buffer = lines.pop() ?? ""
      for (const line of lines) yield line.endsWith("\r") ? line.slice(0, -1) : line
    }
  } finally {
    reader.releaseLock()
  }
}

export function createSSEStream(
  path: string,
  body: Record<string, unknown>,
  base: string = API_BASE
): SSEStreamResult {
  const controller = new AbortController()

  const promise = fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: controller.signal,
  })

  async function* iterate(): AsyncGenerator<SSEEvent> {
    const res = await promise
    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || `HTTP ${res.status}`)
    }

    let currentEvent = "message"
    let eventData: string[] = []

    for await (const line of readLines(res)) {
      if (!line) {
        if (eventData.length > 0) {
          yield { event: currentEvent, data: eventData.join("\n") }
          eventData = []
          currentEvent = "message"
        }
        continue
      }
      if (line.startsWith("event:")) {
        currentEvent = line.slice(6).trim()
        continue
      }
      if (line.startsWith("data:")) {
        const d = line.startsWith("data: ") ? line.slice(6) : line.slice(5)
        eventData.push(d)
      }
    }
    if (eventData.length > 0) {
      yield { event: currentEvent, data: eventData.join("\n") }
    }
  }

  return {
    abort: () => controller.abort(),
    [Symbol.asyncIterator]: () => iterate(),
  }
}

export async function* readSSEEvents(
  stream: SSEStreamResult
): AsyncGenerator<SSEEvent> {
  for await (const event of stream) {
    yield event
  }
}