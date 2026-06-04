import type { Attachment, Message } from "@/lib/types"
import { parseFiles, processStudyPdf } from "@/lib/api"

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
) {
  const pdfAtts = attachments.filter(a => a.mime === "application/pdf")
  const pdfNames = pdfAtts.map(a => a.name)

  setMessages(prev => [
    ...prev,
    { role: "user" as const, content: text || `Study: ${pdfNames.join(", ")}`, attachments },
    { role: "assistant" as const, content: "" },
  ])

  try {
    const parsed = await parseFiles(chatId, pdfNames)
    const enriched = attachments.map(a => {
      if (a.mime !== "application/pdf") return a
      const p = parsed.find(r => r.name === a.name)
      return { ...a, parsedMd: p?.parsedMd ?? undefined }
    })

    setMessages(prev => {
      const copy = [...prev]
      const idx = copy.length - 2
      if (idx >= 0 && copy[idx].role === "user") copy[idx] = { ...copy[idx], attachments: enriched }
      return copy
    })

    const md = enriched.filter(a => a.parsedMd).map(a => `### ${a.name}\n\n${a.parsedMd}`).join("\n\n")

    if (!md.trim()) {
      updateLastMsg(setMessages, m => ({ ...m, content: "Failed to extract content from the PDF." }))
      return
    }

    const result = await processStudyPdf({ markdown: md, filename: pdfNames[0], model })
    const goalsMd = result.topics.map(t => `- [ ] ${t.label}`).join("\n")

    updateLastMsg(setMessages, m => ({
      ...m,
      content: result.overview + "\n\n" + result.article,
      attachments: [{ name: "Learning Goals", mime: "text/markdown", url: "", content: goalsMd }],
    }))
  } catch (e) {
    updateLastMsg(setMessages, m => ({ ...m, content: `Error: ${(e as Error).message || "Study processing failed"}` }))
  }
}