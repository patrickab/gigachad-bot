import type { Attachment, Message } from "./types"

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
