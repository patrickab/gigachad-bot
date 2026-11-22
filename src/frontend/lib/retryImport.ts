// Lazy chunk loading that survives a rebuild: while the dev server recompiles (or a deploy swaps chunks) a chunk
// request can fail once and succeed moments later. Webpack forgets a failed chunk, so calling `load` again refetches it.

const ATTEMPTS = 5
const FIRST_DELAY_MS = 300

/** `load()`, called again with backoff when it rejects, up to `ATTEMPTS` times. Rejects with the last error. */
export async function retryImport<T>(load: () => Promise<T>): Promise<T> {
  for (let attempt = 1, delay = FIRST_DELAY_MS; ; attempt += 1, delay *= 2) {
    try {
      return await load()
    } catch (error) {
      if (attempt >= ATTEMPTS) throw error
    }
    await new Promise((resolve) => setTimeout(resolve, delay))
  }
}
