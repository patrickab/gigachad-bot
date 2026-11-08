import "@testing-library/jest-dom/vitest"
import { afterEach } from "vitest"

// Open tabs persist in sessionStorage, one test's tabs must not restore into the next
afterEach(() => { if (typeof sessionStorage !== "undefined") sessionStorage.clear() })

if (!Promise.withResolvers) {
  Promise.withResolvers = <T>() => {
    let resolve!: (value: T | PromiseLike<T>) => void
    let reject!: (reason?: unknown) => void
    const promise = new Promise<T>((resolvePromise, rejectPromise) => {
      resolve = resolvePromise
      reject = rejectPromise
    })
    return { promise, resolve, reject }
  }
}
