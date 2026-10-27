import type { Attachment } from "@/lib/domain/types"
import { attachRootFile, loadFileViewerText, writeArchitectureSource, writeDocument, writeRootFile } from "./api"
import { fileViewerRawUrl } from "./urls"
import { baseName, extOf, kindOf } from "@/lib/paths"
import { architectureSlug, architectureSource, isArchitecturePath } from "@/lib/domain/architecturePaths"

/** MIME by extension for app documents, matching the backend's guess for the kinds the chat tells apart. */
function documentMime(path: string): string {
  const kind = kindOf(path)
  if (kind === "image") return `image/${extOf(path)}`
  if (kind === "pdf") return "application/pdf"
  return kind === "markdown" ? "text/markdown" : "text/plain"
}

/**
 * Attach any document as a live reference, re-read on every send.
 *
 * - Mount files (absolute paths) and library PDFs (`PDFs/<name>`) go through the file roots, which follow moved files
 * - App documents (logical keys) are read by key, text only
 */
export async function attachReference(path: string): Promise<Attachment> {
  if (path.startsWith("/") || path.startsWith("PDFs/")) return attachRootFile(path)
  const mime = documentMime(path)
  const content = mime.startsWith("text/") ? await loadFileViewerText(path) : undefined
  return { name: baseName(path), mime, url: fileViewerRawUrl(path), active: true, vaultPath: path, content }
}

/**
 * Save `content` to the source of the text reference `path` (an `Attachment.vaultPath`).
 *
 * - Mount files in place, project documents (`project/<slug>/document/<name>`) and loose notes (`note/<name>`) by name, architecture sources in their workspace
 * - Throws for sources that are not text documents (PDFs, images), so a text edit never overwrites them
 */
export async function writeReference(path: string, content: string): Promise<void> {
  if (!documentMime(path).startsWith("text/")) throw new Error(`Not a text document: ${path}`)
  if (path.startsWith("/")) return writeRootFile(path, content)
  if (isArchitecturePath(path)) {
    await writeArchitectureSource(architectureSlug(path), architectureSource(path), content)
    return
  }
  const projectDocument = /^project\/([^/]+)\/document\/([^/]+)$/.exec(path)
  if (projectDocument) {
    await writeDocument(projectDocument[1], projectDocument[2], content)
    return
  }
  const note = /^note\/([^/]+)$/.exec(path)
  if (note) {
    await writeDocument("", note[1], content)
    return
  }
  throw new Error(`Not a text document: ${path}`)
}
