import type { GrammarState, HighlighterCore } from "shiki/core"

type Shiki = { highlighter: HighlighterCore; hastToHtml: typeof import("shiki/core").hastToHtml }

let shikiPromise: Promise<Shiki> | null = null

/** Lazily loads Shiki and creates the shared highlighter. The JS regex engine avoids shipping WASM. */
function getShiki(): Promise<Shiki> {
  shikiPromise ??= import("./shikiBundle.ts").then(async ({ createHighlighterCore, createJavaScriptRegexEngine, hastToHtml, themes, langs }) => ({
    highlighter: await createHighlighterCore({ themes, langs, engine: createJavaScriptRegexEngine() }),
    hastToHtml,
  }))
  return shikiPromise
}

const LANG_ALIASES: Record<string, string> = {
  js: "javascript",
  ts: "typescript",
  py: "python",
  sh: "bash",
  shell: "bash",
  yml: "yaml",
  md: "markdown",
  rb: "ruby",
  rs: "rust",
  kt: "kotlin",
  c4: "likec4",
}

const SHIKI_OPTIONS = {
  themes: { dark: "andromeeda", light: "snazzy-light" },
  defaultColor: false,
} as const

/** Dual-theme HTML (CSS vars per scheme). Unknown languages fall back to plain text. Returns "" if highlighting fails. */
export async function highlightCode(code: string, lang: string): Promise<string> {
  const resolved = LANG_ALIASES[lang] ?? lang
  const { highlighter: hl } = await getShiki()
  const safeLang = hl.getLoadedLanguages().includes(resolved) ? resolved : "text"
  try {
    return hl.codeToHtml(code, { lang: safeLang, ...SHIKI_OPTIONS })
  } catch {
    try {
      return hl.codeToHtml(code, { lang: "text", ...SHIKI_OPTIONS })
    } catch {
      return ""
    }
  }
}

/** Highlighted code as `CodeBlock` mounts it: Shiki's `<pre>` attributes, the markup of the settled lines and of the last line. */
export interface HighlightedLines {
  preClass: string
  preStyle: Record<string, string>
  /** `.line` spans of every line but the last, unchanged until the next line break. */
  settled: string
  /** The `.line` span of the last line. */
  tail: string
}

export interface IncrementalHighlighter {
  /** Highlight `code`. Null if highlighting fails. Calls resolve in the order made. */
  highlight(code: string): Promise<HighlightedLines | null>
}

const lineMarkup = (html: string) => html.slice(html.indexOf("<code>") + 6, html.lastIndexOf("</code>"))

function preAttributes(html: string): Pick<HighlightedLines, "preClass" | "preStyle"> {
  const [, preClass = "shiki", style = ""] = /^<pre class="([^"]*)" style="([^"]*)"/.exec(html) ?? []
  const preStyle: Record<string, string> = {}
  for (const declaration of style.split(";")) {
    const colon = declaration.indexOf(":")
    if (colon > 0) preStyle[declaration.slice(0, colon).trim()] = declaration.slice(colon + 1).trim()
  }
  return { preClass, preStyle }
}

/**
 * A highlighter for code that grows at its end, a streamed fence: each line is tokenized once, when the line after it starts,
 * and only the last line again per call. The markup equals `highlightCode`'s, split at the last line.
 *
 * - The grammar's state at the end of the settled lines is kept, the next lines continue from it (TextMate grammars work line by line)
 * - Code that is no continuation of the last call's starts over
 * - Unknown languages fall back to plain text, as `highlightCode` does
 */
export function createIncrementalHighlighter(lang: string): IncrementalHighlighter {
  const resolved = LANG_ALIASES[lang] ?? lang
  /** The settled lines' text, null before the first line break. */
  let settledCode: string | null = null
  let settled = ""
  let state: GrammarState | undefined

  return {
    async highlight(code) {
      const { highlighter: hl, hastToHtml } = await getShiki()
      let safeLang = hl.getLoadedLanguages().includes(resolved) ? resolved : "text"
      const chunk = (text: string, grammarState: GrammarState | undefined) => {
        const hast = hl.codeToHast(text, { lang: safeLang, ...SHIKI_OPTIONS, grammarState })
        return { html: hastToHtml(hast), state: hl.getLastGrammarState(hast) }
      }
      const startOver = () => {
        settledCode = null
        settled = ""
        state = undefined
      }
      const run = (): HighlightedLines => {
        const lastBreak = code.lastIndexOf("\n")
        const complete = lastBreak < 0 ? null : code.slice(0, lastBreak)
        if (settledCode !== null && (complete === null || !(complete === settledCode || complete.startsWith(`${settledCode}\n`)))) startOver()
        if (complete !== null && complete !== settledCode) {
          const added = chunk(settledCode === null ? complete : complete.slice(settledCode.length + 1), state)
          settled = settledCode === null ? lineMarkup(added.html) : `${settled}\n${lineMarkup(added.html)}`
          state = added.state
          settledCode = complete
        }
        const last = chunk(code.slice(lastBreak + 1), state).html
        return { ...preAttributes(last), settled, tail: lineMarkup(last) }
      }
      try {
        return run()
      } catch {
        startOver()
        safeLang = "text"
        try {
          return run()
        } catch {
          startOver()
          return null
        }
      }
    },
  }
}
