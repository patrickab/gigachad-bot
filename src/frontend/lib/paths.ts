/** Return the last segment of a slash-separated path. */
export function baseName(path: string): string {
  const i = path.lastIndexOf("/")
  return i === -1 ? path : path.slice(i + 1)
}

/** Return everything before the last slash, or "" when there is none. */
export function dirName(path: string): string {
  const i = path.lastIndexOf("/")
  return i === -1 ? "" : path.slice(0, i)
}

const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "avif", "ico"])

export type FileKind = "image" | "pdf" | "markdown" | "text"

/** Lowercase extension of the file name without the dot, "" for none (dotfiles like ".env" have none). */
export function extOf(path: string): string {
  const name = baseName(path)
  const i = name.lastIndexOf(".")
  return i <= 0 ? "" : name.slice(i + 1).toLowerCase()
}

/** How a file is shown: image and PDF from the raw URL, markdown rendered, anything else as text. */
export function kindOf(path: string): FileKind {
  const ext = extOf(path)
  if (IMAGE_EXTENSIONS.has(ext)) return "image"
  if (ext === "pdf") return "pdf"
  if (ext === "md" || ext === "markdown") return "markdown"
  return "text"
}
