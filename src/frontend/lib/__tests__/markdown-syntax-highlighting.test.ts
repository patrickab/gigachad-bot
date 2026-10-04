import { describe, expect, it } from "vitest"
import { highlightCode } from "@/lib/markdown-syntax-highlighting"

describe("highlightCode", () => {
  // An unknown or unloadable grammar falls back to plain text without an error,
  // so a broken LikeC4 grammar would only show up as an uncoloured editor.
  it("tokenizes LikeC4 sources, also under the c4 fence alias", async () => {
    const source = "model {\n  shop = system 'Shop'\n}\n"
    const plain = await highlightCode(source, "text")
    for (const lang of ["likec4", "c4"]) {
      const html = await highlightCode(source, lang)
      expect(html).not.toBe(plain)
      expect(html).toMatch(/--shiki-dark:[^;"]+/)
    }
  })
})
