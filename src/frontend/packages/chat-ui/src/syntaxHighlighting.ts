import type { HighlighterCore } from "shiki/core"

let highlighterPromise: Promise<HighlighterCore> | null = null

/** Lazily loads Shiki and creates the shared highlighter. The JS regex engine avoids shipping WASM. */
function getHighlighter(): Promise<HighlighterCore> {
  highlighterPromise ??= import("./shikiBundle.ts").then(({ createHighlighterCore, createJavaScriptRegexEngine, themes, langs }) =>
    createHighlighterCore({ themes, langs, engine: createJavaScriptRegexEngine() }))
  return highlighterPromise
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
  const hl = await getHighlighter()
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