// Chat upload routes under /files.
import { chatFileUrl, toQuery } from "../urls"
import { del, fileForm, request } from "./http"
import type { Attachment } from "@/lib/domain/types"

type AttachmentPayload = Omit<Attachment, "active" | "url">

function toChatAttachment(data: AttachmentPayload, chatId: string, slug: string | null): Attachment {
  return { ...data, active: true, url: chatFileUrl(chatId, data.name, slug) }
}

export async function uploadFile(chatId: string, file: File, slug: string | null = null, overwrite = false): Promise<Attachment> {
  const query = toQuery({ chat_id: chatId, slug, overwrite })
  return toChatAttachment(await request<AttachmentPayload>(`/files/upload${query}`, { method: "POST", body: fileForm(file) }), chatId, slug)
}

interface ParsedAttachment {
  name: string
  parsedMd: string | null
}

export async function parseFiles(chatId: string, filenames: string[], slug: string | null = null): Promise<ParsedAttachment[]> {
  const query = toQuery({ chat_id: chatId, filenames, slug })
  return request(`/files/parse${query}`, { method: "POST" })
}

export async function deleteAttachment(chatId: string, filename: string, slug: string | null = null): Promise<void> {
  await del(`/files/chat/${chatId}/att/${encodeURIComponent(filename)}${toQuery({ slug })}`)
}
