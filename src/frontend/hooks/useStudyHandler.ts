import type { Attachment, Message } from "@/lib/types"
import { parseFiles, processStudyPdf, registerUploadAsDocument } from "@/lib/api"

export function updateLastMsg(
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>,
  updater: (msg: Message) => Message,
) {
  setMessages(prev => {
    const copy = [...prev]
    const last = copy[copy.length - 1]
    if (last?.role === "assistant") copy[copy.length - 1] = updater(last)
    return copy
  })
}

export async function handleStudyPdf(
  text: string,
  attachments: Attachment[],
  chatId: string,
  model: string,
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>,
  slug: string | null = null,
) {
  const pdfAtts = attachments.filter(a => a.mime === "application/pdf")
  const pdfNames = pdfAtts.map(a => a.name)

  setMessages(prev => [
    ...prev,
    { role: "user" as const, content: text || `Study: ${pdfNames.join(", ")}`, attachments: attachments.map((a) => ({ ...a, active: true })) },
    { role: "assistant" as const, content: "Extracting document content…" },
  ])

  try {
    const needsParsing = pdfAtts.filter(a => !a.parsedMd && !a.content)
    let enriched = attachments

    if (needsParsing.length > 0) {
      const parsed = await parseFiles(chatId, needsParsing.map(a => a.name), slug)
      enriched = attachments.map(a => {
        if (a.parsedMd || a.content || a.mime.startsWith("image/")) return a
        const p = parsed.find(r => r.name === a.name)
        return { ...a, parsedMd: p?.parsedMd ?? undefined }
      })

      setMessages(prev => {
        const copy = [...prev]
        const idx = copy.length - 2
        if (idx >= 0 && copy[idx].role === "user") copy[idx] = { ...copy[idx], attachments: enriched }
        return copy
      })
    }

    for (const att of pdfAtts) {
      registerUploadAsDocument(chatId, att.name, slug).catch(() => {})
    }

    const md = enriched.filter(a => a.parsedMd).map(a => `### ${a.name}\n\n${a.parsedMd}`).join("\n\n")

    if (!md.trim()) {
      updateLastMsg(setMessages, m => ({ ...m, content: "Failed to extract content from the PDF." }))
      return
    }

    updateLastMsg(setMessages, m => ({ ...m, content: "Generating study materials…" }))

    const result = await processStudyPdf({ markdown: md, filename: pdfNames[0], model })

    updateLastMsg(setMessages, m => ({
      ...m,
      content: result.mindmap + "\n\n" + result.overview + "\n\n" + result.article,
    }))
  } catch (e) {
    updateLastMsg(setMessages, m => ({ ...m, content: `Error: ${(e as Error).message || "Study processing failed"}` }))
  }
}