import { getApiBase } from "./config"

type QueryValue = string | number | boolean | string[] | undefined | null

export function encodePath(filename: string): string {
  return filename.split("/").map(encodeURIComponent).join("/")
}

/** Omits null, undefined and false. Arrays repeat the key. true becomes "true". */
export function toQuery(params: Record<string, QueryValue>): string {
  const usp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === false) continue
    if (Array.isArray(v)) v.forEach((item) => usp.append(k, item))
    else usp.set(k, v === true ? "true" : String(v))
  }
  const s = usp.toString()
  return s ? `?${s}` : ""
}

/** Origin of the backend, for asset URLs the backend emits as root-relative paths. */
export function apiOrigin(): string {
  return new URL(getApiBase()).origin
}

// Asset URLs use database-native namespaces. Local filesystem layout is never
// part of the persisted attachment identity.
function uploadsBase(chatId: string, slug: string | null): string {
  const chat = encodeURIComponent(chatId)
  const prefix = slug
    ? `attachment/project/${encodeURIComponent(slug)}/chat/${chat}`
    : `attachment/chat/${chat}`
  return `${getApiBase()}/assets/${prefix}`
}

/** Point MinerU's relative `images/<name>` markdown refs at the per-user library cache, keyed by the PDF's stem. */
export function rewriteImages(content: string, pdfName: string): string {
  const stem = encodeURIComponent(pdfName.replace(/\.pdf$/i, ""))
  return content.replace(/\(images\/([^)]+)\)/g, `(${getApiBase()}/assets/Mineru/images/${stem}/$1)`)
}

export function chatFileUrl(chatId: string, filename: string, slug: string | null = null): string {
  return `${uploadsBase(chatId, slug)}/${encodeURIComponent(filename)}`
}

/** Direct URL to a file's raw bytes (images, PDFs), usable as an `<img>`/PDF src. */
export function fileViewerRawUrl(path: string): string {
  return `${getApiBase()}/fileviewer/raw?path=${encodeURIComponent(path)}`
}

export function sandboxAssetUrl(assetPath: string): string {
  return `${getApiBase()}/assets/${encodePath(assetPath)}`
}

export function canvasMutationStreamUrl(canvasKey: string, sinceRevision: number): string {
  return `${getApiBase()}/canvases/${encodePath(canvasKey)}/stream${toQuery({ sinceRevision })}`
}
