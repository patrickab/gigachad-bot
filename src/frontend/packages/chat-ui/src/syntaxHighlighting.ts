import { createHighlighterCore, type HighlighterCore, type LanguageRegistration } from "shiki/core"
import { createJavaScriptRegexEngine } from "shiki/engine/javascript"
import andromeeda from "shiki/themes/andromeeda.mjs"
import snazzyLight from "shiki/themes/snazzy-light.mjs"
import javascript from "shiki/langs/javascript.mjs"
import typescript from "shiki/langs/typescript.mjs"
import python from "shiki/langs/python.mjs"
import bash from "shiki/langs/bash.mjs"
import sql from "shiki/langs/sql.mjs"
import json from "shiki/langs/json.mjs"
import html from "shiki/langs/html.mjs"
import css from "shiki/langs/css.mjs"
import yaml from "shiki/langs/yaml.mjs"
import markdown from "shiki/langs/markdown.mjs"
import shellscript from "shiki/langs/shellscript.mjs"
import tsx from "shiki/langs/tsx.mjs"
import jsx from "shiki/langs/jsx.mjs"
import java from "shiki/langs/java.mjs"
import c from "shiki/langs/c.mjs"
import cpp from "shiki/langs/cpp.mjs"
import go from "shiki/langs/go.mjs"
import rust from "shiki/langs/rust.mjs"
import ruby from "shiki/langs/ruby.mjs"
import php from "shiki/langs/php.mjs"
import swift from "shiki/langs/swift.mjs"
import kotlin from "shiki/langs/kotlin.mjs"
import r from "shiki/langs/r.mjs"
import latex from "shiki/langs/latex.mjs"
import xml from "shiki/langs/xml.mjs"
import dockerfile from "shiki/langs/dockerfile.mjs"
import toml from "shiki/langs/toml.mjs"
import ini from "shiki/langs/ini.mjs"
import diff from "shiki/langs/diff.mjs"
// Shiki ships no LikeC4 grammar: this is the VS Code extension's TextMate grammar,
// vendored unchanged from likec4/likec4@v1.59.4 packages/vscode (MIT, (c) Denis Davydkov),
// pinned to the LikeC4 version gigachad-bot's src/c4 parses with. The import attribute lets Node load it from the package.
import likec4 from "./grammars/likec4.tmLanguage.json" with { type: "json" }

let highlighterPromise: Promise<HighlighterCore> | null = null

/** Lazily creates the shared highlighter. The JS regex engine avoids shipping WASM. */
function getHighlighter(): Promise<HighlighterCore> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighterCore({
      themes: [andromeeda, snazzyLight],
      langs: [
        javascript,
        typescript,
        python,
        bash,
        sql,
        json,
        html,
        css,
        yaml,
        markdown,
        shellscript,
        tsx,
        jsx,
        java,
        c,
        cpp,
        go,
        rust,
        ruby,
        php,
        swift,
        kotlin,
        r,
        latex,
        xml,
        dockerfile,
        toml,
        ini,
        diff,
        likec4 as unknown as LanguageRegistration,
      ],
      engine: createJavaScriptRegexEngine(),
    })
  }
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