import type { Attachment, ChatRequest, Message, ToolCallRecord } from "./types"
import { apiOrigin, chatFileUrl, fileViewerRawUrl } from "./urls"
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

const ATTACHMENT_ONLY_PROMPT = "Please review the attached document and provide a summary."

/** The text the model receives for a user turn: attached-file content ahead of the prompt, with a default prompt for attachment-only turns. */
export function composeUserMessage(text: string, hiddenContent: string | undefined): string {
  const prompt = text.trim() ? text : ATTACHMENT_ONLY_PROMPT
  return hiddenContent ? `${hiddenContent}\n\n${prompt}` : prompt
}

export function collectActiveImagePaths(
  messages: Message[],
  scope?: { userIndex: number },
): string[] {
  const paths: string[] = []
  if (scope) {
    const msg = messages[scope.userIndex]
    for (const a of msg?.attachments ?? []) {
      if (a.active && isImageAttachment(a)) paths.push(a.name)
    }
    return paths
  }
  for (const m of messages) {
    if (m.role !== "user" || !m.attachments) continue
    for (const a of m.attachments) {
      if (a.active && isImageAttachment(a)) paths.push(a.name)
    }
  }
  return paths
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
  parse: (chatId: string, names: string[], slug: string | null) => Promise<{ name: string; parsedMd: string | null }[]>,
  refreshRootFile?: (path: string) => Promise<Attachment>,
): Promise<{ enriched: Attachment[]; request: ChatRequest }> {
  const { text, attachments, priorMessages, chatId, slug, baseParams } = opts

  // File root refs are live: re-read content from the source at send time (also
  // picks up a PDF extraction that finished after attach). No upload copy
  // exists, so they never go through the chat-upload parse endpoint.
  let enriched = attachments
  if (refreshRootFile) {
    enriched = await Promise.all(enriched.map(async (a) => {
      if (!a.vaultPath) return a
      try {
        const fresh = await refreshRootFile(a.vaultPath)
        return { ...a, content: fresh.content, parsedMd: fresh.parsedMd ?? a.parsedMd }
      } catch {
        return a // dangling ref (deleted file / unmounted file root): keep last known content
      }
    }))
  }

  const needsParsing = enriched.filter((a) => !isImageAttachment(a) && !a.vaultPath && !a.content && !a.parsedMd)
  if (needsParsing.length > 0) {
    const parsed = await parse(chatId, needsParsing.map((a) => a.name), slug)
    enriched = enriched.map((a) => {
      if (a.content || a.parsedMd || a.vaultPath || isImageAttachment(a)) return a
      const p = parsed.find((r) => r.name === a.name)
      return { ...a, parsedMd: p?.parsedMd ?? undefined }
    })
  }

  const priorImgPaths = collectActiveImagePaths(priorMessages)
  const newImgPaths = enriched.filter((a) => a.active && isImageAttachment(a)).map((a) => a.name)

  const request: ChatRequest = {
    ...baseParams,
    chat_id: chatId,
    user_msg: composeUserMessage(text, buildHiddenContent(enriched) || undefined),
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
      if (sent.has(a.name) && isImageAttachment(a) && a.active) {
        changed = true
        return { ...a, active: false }
      }
      return a
    })
    return changed ? { ...m, attachments } : m
  })
}
