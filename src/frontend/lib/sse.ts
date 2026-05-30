export interface SSEStreamResult {
  stream: ReadableStreamDefaultReader<Uint8Array>
  abort: () => void
}

export function createSSEStream(
  path: string,
  body: Record<string, unknown>,
  base: string = "/api"
): SSEStreamResult {
  const controller = new AbortController()

  const promise = fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: controller.signal,
  })

  return {
    stream: new ReadableStream({
      async start(ctrl) {
        const res = await promise
        if (!res.ok || !res.body) {
          ctrl.error(await res.text())
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
              ctrl.enqueue(new TextEncoder().encode(eventData.join("\n")))
            }
            ctrl.close()
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