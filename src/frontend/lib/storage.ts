/** `localStorage.getItem(key)`, null when storage is unavailable. */
export function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

/** `localStorage.setItem(key, value)`, silently skipped when storage is unavailable or full. */
export function writeStorage(key: string, value: string): void {
  try {
    localStorage.setItem(key, value)
  } catch {
    // Unavailable or over quota
  }
}
