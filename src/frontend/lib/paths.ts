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

/** `path` with a home directory shown as `~`. */
export function shortPath(path: string): string {
  return path.replace(/^\/(?:home|Users)\/[^/]+(?=\/|$)/, "~")
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

/** Whether `path` is a notebook the notebook editor opens: an absolute `.ipynb` (kernels and saves need the host path). */
export function isNotebookPath(path: string): boolean {
  return path.startsWith("/") && path.endsWith(".ipynb")
}

/** Natural, case-insensitive order for paths and file names ("file2" before "file10"). */
export const PATH_COLLATOR = new Intl.Collator("en", { numeric: true, sensitivity: "base" })

/** `path` without trailing slashes, the root `/` stays. */
export function trimSlash(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path
}

/** Whether `path` is `root` or below it, `/a/foobar` is not inside `/a/foo`. */
export function isInside(path: string, root: string): boolean {
  const [inner, outer] = [trimSlash(path), trimSlash(root)]
  return inner === outer || inner.startsWith(outer === "/" ? "/" : `${outer}/`)
}

/** `path` relative to the directory `base` when below it, else unchanged. */
export function relativeIn(base: string, path: string): string {
  const prefix = base.endsWith("/") ? base : `${base}/`
  return path.startsWith(prefix) ? path.slice(prefix.length) : path
}

/** `path` relative to the directory `folder`: `chapters/intro.tex`, or `../main.tex` when it lies above or beside the folder. */
export function relativeTo(folder: string, path: string): string {
  const from = folder.split("/").filter(Boolean)
  const to = path.split("/").filter(Boolean)
  let shared = 0
  // The file's own name never counts as a shared folder
  while (shared < from.length && shared < to.length - 1 && from[shared] === to[shared]) shared++
  return [...from.slice(shared).map(() => ".."), ...to.slice(shared)].join("/")
}
