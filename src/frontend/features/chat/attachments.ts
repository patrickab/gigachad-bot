import type { Attachment, ChatRequest, Message, ToolCallRecord } from "@/lib/domain/types"
import { apiOrigin, chatFileUrl, fileViewerRawUrl } from "@/lib/backend/urls"
import { ApiError } from "@/lib/backend/api"
import { writeReference } from "@/lib/backend/references"
import { toolPayload } from "./toolCalls"


interface ToolArtifactDocument {
  name: string
  content: string
}

function artifactName(call: ToolCallRecord, extension: string): string {
  const id = call.id.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "generated"
  return `${call.name}-${id}${extension}`
}

/** Turn a renderable chat-tool result into a normal document for later reuse. */
export function toolArtifactDocument(call: ToolCallRecord): ToolArtifactDocument | null {
  const payload = toolPayload(call)
  if (payload.family === "diagram" && payload.mermaid !== null) {
    return {
      name: artifactName(call, ".md"),
      content: `# Mermaid diagram\n\n\`\`\`mermaid\n${payload.mermaid}\n\`\`\`\n`,
    }
  }
  if (payload.family === "mindmap" && payload.content !== null) {
    return { name: artifactName(call, ".md"), content: payload.content }
  }
  if (payload.family === "plot" && payload.figure) {
    return { name: artifactName(call, ".plot.json"), content: JSON.stringify(payload.figure, null, 2) }
  }
  return null
}

export function isImageAttachment(a: Attachment): boolean {
  return a.mime.startsWith("image/")
}

const isActiveImage = (a: Attachment) => a.active && isImageAttachment(a)

// Stored `url`s are absolute against whatever API origin was active at upload time
// (e.g. an old localhost dev backend), so trusting them breaks loads on other origins.
// Path+query is the only reliable locator: chat_id/slug are NOT stable, since uploads
// stay under the project active at send time even when a chat is moved or branched.
// Rebase onto the current origin, and rebuild from (chatId, slug) only without a stored URL.
function rebaseOnApiOrigin(url: string): string | null {
  try {
    const u = new URL(url)
    return `${apiOrigin()}${u.pathname}${u.search}`
  } catch {
    return null
  }
}

export function normalizeAttachment(a: Attachment, chatId: string | null, slug: string | null): Attachment {
  const active = a.active !== undefined ? a.active : !isImageAttachment(a)
  const rebased = a.url ? rebaseOnApiOrigin(a.url) : null
  const url = rebased ?? (a.vaultPath ? fileViewerRawUrl(a.vaultPath) : chatId ? chatFileUrl(chatId, a.name, slug) : a.url)
  return { ...a, active, url }
}

export function normalizeMessageAttachments(msg: Message, chatId: string | null, slug: string | null): Message {
  if (!msg.attachments?.length) return msg
  return withAttachments(msg, msg.attachments.map((a) => normalizeAttachment(a, chatId, slug)))
}

export function buildHiddenContent(attachments: Attachment[]): string {
  const textParts: string[] = []
  for (const a of attachments) {
    if (!a.active) continue
    if (isImageAttachment(a)) continue
    const content = a.parsedMd ?? a.content
    if (content) textParts.push(`### ${a.name}\n\n${content}`)
  }
  return textParts.length > 0
    ? "**Attached files:**\n\n" + textParts.join("\n\n") + "\n\n## END"
    : ""
}

/** Replace a message's attachments and rebuild the model-facing `hiddenContent` from them. */
export function withAttachments(msg: Message, attachments: Attachment[]): Message {
  return { ...msg, attachments, hiddenContent: buildHiddenContent(attachments) || undefined }
}

/** The attachment of `msg` named `name`, if any. */
export function findAttachment(msg: Message | undefined, name: string): Attachment | undefined {
  return msg?.attachments?.find((a) => a.name === name)
}

/** `msg` with its attachment `name` replaced by `update(attachment)` (null drops it) and `hiddenContent` rebuilt; null when `msg` has no attachments. */
export function withAttachment(msg: Message, name: string, update: (attachment: Attachment) => Attachment | null): Message | null {
  if (!msg.attachments) return null
  return withAttachments(msg, msg.attachments.flatMap((a) => a.name === name ? (update(a) ?? []) : [a]))
}

const ATTACHMENT_ONLY_PROMPT = "Please review the attached document and provide a summary."

/** The text the model receives for a user turn: attached-file content ahead of the prompt, with a default prompt for attachment-only turns. */
export function composeUserMessage(text: string, hiddenContent = ""): string {
  const prompt = text.trim() ? text : ATTACHMENT_ONLY_PROMPT
  return hiddenContent ? `${hiddenContent}\n\n${prompt}` : prompt
}

export function collectActiveImagePaths(
  messages: Message[],
  scope?: { userIndex: number },
): string[] {
  const scoped = scope ? [messages[scope.userIndex]] : messages.filter((m) => m.role === "user")
  return scoped.flatMap((m) => (m?.attachments ?? []).filter(isActiveImage).map((a) => a.name))
}

type ParseAttachments = (chatId: string, names: string[], slug: string | null) => Promise<{ name: string; parsedMd: string | null }[]>

/** Parse the uploaded non-image attachments that carry no text yet (PDFs) and merge the markdown in. */
export async function withParsedAttachments(attachments: Attachment[], chatId: string, slug: string | null, parse: ParseAttachments): Promise<Attachment[]> {
  const needsParsing = attachments.filter((a) => !isImageAttachment(a) && !a.vaultPath && !a.content && !a.parsedMd)
  if (needsParsing.length === 0) return attachments
  const parsed = await parse(chatId, needsParsing.map((a) => a.name), slug)
  return attachments.map((a) => {
    if (!needsParsing.includes(a)) return a
    const p = parsed.find((r) => r.name === a.name)
    return { ...a, parsedMd: p?.parsedMd ?? undefined }
  })
}

/** Re-read a document reference by its path (`attachReference` in the API module). */
export type AttachReference = (path: string) => Promise<Attachment>

/** Result per re-read path: the fresh attachment, `"missing"` for a 404, null for any other failure (kept as is, retried next time). */
export type ReferenceReads = Map<string, Attachment | "missing" | null>

/** Paths a send re-reads: active references found last time. Inactive ones never reach the model, a missing one keeps its last known text. */
export function liveReferencePaths(attachments: Attachment[]): string[] {
  return attachments.flatMap((a) => a.vaultPath && a.active && !a.missing ? [a.vaultPath] : [])
}

/** Read each distinct path in `paths` once. */
export async function readReferences(paths: string[], attachReference: AttachReference): Promise<ReferenceReads> {
  const unique = [...new Set(paths)]
  const results = await Promise.all(unique.map((path) => attachReference(path).catch((error) =>
    error instanceof ApiError && error.status === 404 ? "missing" as const : null)))
  return new Map(unique.map((path, i) => [path, results[i]]))
}

/**
 * `attachment` updated from the read of its source, itself when it was not read or the read failed.
 *
 * - Follows a moved file: the fresh `vaultPath` and `url` replace the old ones
 * - A missing source keeps the last known content and is marked `missing`. A PDF then shows its PDF-library copy
 * - A PDF whose extraction is still running keeps its previous extraction
 */
export function withReferenceRead(attachment: Attachment, reads: ReferenceReads): Attachment {
  const read = attachment.vaultPath ? reads.get(attachment.vaultPath) : null
  if (read == null) return attachment
  if (read === "missing") {
    const url = attachment.mime === "application/pdf" ? fileViewerRawUrl(`PDFs/${attachment.name}`) : attachment.url
    return { ...attachment, url, missing: true }
  }
  return { ...attachment, content: read.content, parsedMd: read.parsedMd ?? attachment.parsedMd, vaultPath: read.vaultPath, url: read.url, missing: undefined }
}

/** `messages` with every reference in `reads` updated and the text the model reads rebuilt. Untouched messages keep their identity. */
export function applyReferenceReads(messages: Message[], reads: ReferenceReads): Message[] {
  if (reads.size === 0) return messages
  return messages.map((m) => m.attachments?.some((a) => a.vaultPath && reads.has(a.vaultPath))
    ? withAttachments(m, m.attachments.map((a) => withReferenceRead(a, reads)))
    : m)
}

/**
 * `attachment` with `text` as its edited text (`parsedMd` wins over `content`), saved to its source first when it is a reference.
 *
 * - A missing reference is a read-only snapshot: saving it would recreate the deleted file
 */
export async function saveAttachmentEdit(attachment: Attachment, text: string): Promise<Attachment> {
  if (attachment.missing) throw new Error(`${attachment.name} no longer exists`)
  if (attachment.vaultPath) await writeReference(attachment.vaultPath, text)
  return attachment.parsedMd != null ? { ...attachment, parsedMd: text } : { ...attachment, content: text }
}

// Pure attachment-intake transform: parses unparsed non-image attachments (via injected
// `parse`), merges results back, and assembles the ChatRequest with prior + new image
// paths. The optimistic placeholder and post-parse patch stay in the caller. Throws if
// `parse` throws, and the caller renders the failure.
export async function buildAttachedSend(
  opts: {
    text: string
    attachments: Attachment[] // already marked active
    priorMessages: Message[]
    chatId: string
    slug: string | null
    baseParams: Omit<ChatRequest, "chat_id" | "user_msg" | "img_paths" | "project_slug">
  },
  parse: ParseAttachments,
): Promise<{ enriched: Attachment[]; request: ChatRequest }> {
  const { text, attachments, priorMessages, chatId, slug, baseParams } = opts

  const enriched = await withParsedAttachments(attachments, chatId, slug, parse)

  const priorImgPaths = collectActiveImagePaths(priorMessages)
  const newImgPaths = enriched.filter(isActiveImage).map((a) => a.name)

  const request: ChatRequest = {
    ...baseParams,
    chat_id: chatId,
    user_msg: composeUserMessage(text, buildHiddenContent(enriched)),
    img_paths: [...priorImgPaths, ...newImgPaths],
    project_slug: slug,
  }

  return { enriched, request }
}

export function deactivateSentImages(messages: Message[], sentPaths: string[]): Message[] {
  if (sentPaths.length === 0) return messages
  const sent = new Set(sentPaths)
  return messages.map((m) => {
    if (m.role !== "user" || !m.attachments) return m
    let changed = false
    const attachments = m.attachments.map((a) => {
      if (sent.has(a.name) && isActiveImage(a)) {
        changed = true
        return { ...a, active: false }
      }
      return a
    })
    return changed ? { ...m, attachments } : m
  })
}
