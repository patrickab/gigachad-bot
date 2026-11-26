import { describe, expect, it } from "vitest"
import { createIncrementalHighlighter, highlightCode } from "../src/syntaxHighlighting"

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

describe("createIncrementalHighlighter", () => {
  const whole = (lines: { settled: string; tail: string } | null) => (lines ? (lines.settled ? `${lines.settled}\n${lines.tail}` : lines.tail) : null)
  const markup = (html: string) => html.slice(html.indexOf("<code>") + 6, html.lastIndexOf("</code>"))

  it("gives a streamed fence the markup of a whole highlight at every length, across multi-line strings", async () => {
    const samples: [string, string][] = [
      ["python", 'def f(x):\n    """doc\n    string"""\n    return f"{x} and \'q\'"\n\nclass A:\n    pass'],
      ["ts", "const a = `line ${1 +\n  2} more\n`\n/* open\n comment */ let b: number = 1"],
      ["nosuchlang", "plain\ntext"],
    ]
    for (const [lang, source] of samples) {
      const highlighter = createIncrementalHighlighter(lang)
      for (let length = 1; length <= source.length; length++) {
        const code = source.slice(0, length)
        expect(whole(await highlighter.highlight(code)), `${lang} at ${length}`).toBe(markup(await highlightCode(code, lang)))
      }
    }
  }, 30_000)

  it("starts over when the code is no continuation, and carries Shiki's pre attributes", async () => {
    const highlighter = createIncrementalHighlighter("python")
    await highlighter.highlight('x = """open\nstill')
    const lines = await highlighter.highlight("y = 1\nz = 2")
    expect(whole(lines)).toBe(markup(await highlightCode("y = 1\nz = 2", "python")))
    expect(lines?.preClass).toContain("shiki")
    expect(lines?.preStyle["--shiki-dark"]).toMatch(/^#/)
  })
})
