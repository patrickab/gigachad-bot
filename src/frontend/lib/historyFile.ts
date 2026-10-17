/** Split `slug/filename` (project chat) into parts. A bare filename has no slug. */
export function parseHistoryFile(historyFile: string): { slug: string | null; filename: string } {
  const parts = historyFile.split("/")
  if (parts.length > 1) {
    return { slug: parts[0], filename: parts.slice(1).join("/") }
  }
  return { slug: null, filename: historyFile }
}

/** Inverse of `parseHistoryFile`. */
export function buildHistoryFile(filename: string, slug: string | null): string {
  return slug ? `${slug}/${filename}` : filename
}
