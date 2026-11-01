import { fileViewerRawUrl } from "@/lib/backend/urls"

/**
 * Run `build` one at a time. The returned trigger resolves once a build that started after the call has finished.
 *
 * - A trigger during a build asks for exactly one more build after it, however many arrive
 * - `build` must not throw
 */
export function createBuildQueue(build: () => Promise<void>): () => Promise<void> {
  let running: Promise<void> | null = null
  let followUp = false
  const drain = async () => {
    try {
      do {
        followUp = false
        await build()
      } while (followUp)
    } finally {
      running = null
    }
  }
  return () => {
    if (running) {
      followUp = true
      return running
    }
    running = drain()
    return running
  }
}

/** Raw URL of the built PDF, `revision` makes the browser and the viewer load the new bytes. */
export function pdfUrlOf(pdfPath: string, revision: number): string {
  return `${fileViewerRawUrl(pdfPath)}&v=${revision}`
}
