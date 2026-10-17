import { getApiBase } from "./config"
import { deviceHeaders } from "./deviceId"

export interface SSEEvent {
  event: string
  data: string
}

export interface SSEStreamResult {
  abort: () => void
  [Symbol.asyncIterator]: () => AsyncIterator<SSEEvent>
}

/** Yields response lines without terminators. A chunk may end mid-line, so the tail is buffered. Handles CRLF. */
async function* readLines(res: Response): AsyncGenerator<string> {
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

/** POST-based SSE (EventSource cannot POST). Aborting cancels the request. Iteration throws on non-2xx. */
export function createSSEStream(path: string, body: Record<string, unknown>): SSEStreamResult {
  const controller = new AbortController()

  const promise = fetch(`${getApiBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...deviceHeaders() },
    body: JSON.stringify(body),
    signal: controller.signal,
  })

  async function* iterate(): AsyncGenerator<SSEEvent> {
    const res = await promise
    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || `HTTP ${res.status}`)
    }

    // Per the SSE spec, an event is dispatched at a blank line. Multiple `data:` lines join with "\n"
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
        // One optional space after the colon is part of the field syntax
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
