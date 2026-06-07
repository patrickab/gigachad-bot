export function extractThought(content: string): { thought: string | null; cleanContent: string } {
  const startTag = "<thought>\n"
  const endTag = "\n</thought>"
  const startIdx = content.indexOf(startTag)
  if (startIdx === -1) return { thought: null, cleanContent: content }

  const endIdx = content.indexOf(endTag, startIdx + startTag.length)
  const before = content.slice(0, startIdx)

  if (endIdx !== -1) {
    const t = content.slice(startIdx + startTag.length, endIdx).trim() || null
    const clean = (before + content.slice(endIdx + endTag.length)).trim()
    return { thought: t, cleanContent: clean }
  }
  const t = content.slice(startIdx + startTag.length).trim() || null
  return { thought: t, cleanContent: before.trim() }
}
