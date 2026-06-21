import type { Attachment, ChatRequest, Message } from "./types"

export function isImageAttachment(a: Attachment): boolean {
  return a.mime.startsWith("image/")
}

export function normalizeAttachment(a: Attachment): Attachment {
  if (a.active !== undefined) return a
  return { ...a, active: !isImageAttachment(a) }
}

export function normalizeMessageAttachments(msg: Message): Message {
  if (!msg.attachments?.length) return msg
  const attachments = msg.attachments.map(normalizeAttachment)
  return {
    ...msg,
    attachments,
    hiddenContent: buildHiddenContent(attachments) || undefined,
  }
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

export function deactivateUserImages(attachments: Attachment[]): Attachment[] {
  return attachments.map((a) => (isImageAttachment(a) ? { ...a, active: false } : a))
}

// Pure attachment-intake transform. Parses unparsed non-image attachments
// (via injected `parse`), merges results back, builds the hidden-content block,
// and assembles the ChatRequest with prior + new image paths. The optimistic
// placeholder message and the post-parse patch stay in the caller — this is the
// testable core lifted out of page.tsx's handleSend. Throws if `parse` throws;
// the caller renders the parse-failure message.
export async function buildAttachedSend(
  opts: {
    text: string
    attachments: Attachment[] // already marked active
    priorMessages: Message[]
    chatId: string
    slug: string | null
    downscaleImages: boolean
    baseParams: Omit<ChatRequest, "chat_id" | "user_msg" | "img_paths" | "project_slug">
  },
  parse: (chatId: string, names: string[], slug: string | null) => Promise<{ name: string; parsedMd: string | null }[]>,
): Promise<{ enriched: Attachment[]; hiddenContent?: string; request: ChatRequest }> {
  const { text, attachments, priorMessages, chatId, slug, downscaleImages, baseParams } = opts

  const needsParsing = attachments.filter((a) => !isImageAttachment(a) && !a.content && !a.parsedMd)
  let enriched = attachments
  if (needsParsing.length > 0) {
    const parsed = await parse(chatId, needsParsing.map((a) => a.name), slug)
    enriched = attachments.map((a) => {
      if (a.content || a.parsedMd || isImageAttachment(a)) return a
      const p = parsed.find((r) => r.name === a.name)
      return { ...a, parsedMd: p?.parsedMd ?? undefined }
    })
  }

  const hiddenContent = buildHiddenContent(enriched) || undefined
  const priorImgPaths = collectActiveImagePaths(priorMessages)
  const newImgPaths = enriched.filter((a) => a.active && isImageAttachment(a)).map((a) => a.name)
  const fallbackPrompt = text.trim() ? text : "Please review the attached document and provide a summary."
  const userMsg = hiddenContent ? `${hiddenContent}\n\n${fallbackPrompt}` : fallbackPrompt

  const request: ChatRequest = {
    ...baseParams,
    chat_id: chatId,
    user_msg: userMsg,
    img_paths: [...priorImgPaths, ...newImgPaths],
    downscale_images: downscaleImages,
    project_slug: slug,
  }

  return { enriched, hiddenContent, request }
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
