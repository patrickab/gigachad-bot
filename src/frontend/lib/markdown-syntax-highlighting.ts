import { createHighlighterCore, type HighlighterCore, type LanguageRegistration } from "shiki/core"
import { createJavaScriptRegexEngine } from "shiki/engine/javascript"
import andromeeda from "@shikijs/themes/andromeeda"
import snazzyLight from "@shikijs/themes/snazzy-light"
import javascript from "@shikijs/langs/javascript"
import typescript from "@shikijs/langs/typescript"
import python from "@shikijs/langs/python"
import bash from "@shikijs/langs/bash"
import sql from "@shikijs/langs/sql"
import json from "@shikijs/langs/json"
import html from "@shikijs/langs/html"
import css from "@shikijs/langs/css"
import yaml from "@shikijs/langs/yaml"
import markdown from "@shikijs/langs/markdown"
import shellscript from "@shikijs/langs/shellscript"
import tsx from "@shikijs/langs/tsx"
import jsx from "@shikijs/langs/jsx"
import java from "@shikijs/langs/java"
import c from "@shikijs/langs/c"
import cpp from "@shikijs/langs/cpp"
import go from "@shikijs/langs/go"
import rust from "@shikijs/langs/rust"
import ruby from "@shikijs/langs/ruby"
import php from "@shikijs/langs/php"
import swift from "@shikijs/langs/swift"
import kotlin from "@shikijs/langs/kotlin"
import r from "@shikijs/langs/r"
import latex from "@shikijs/langs/latex"
import xml from "@shikijs/langs/xml"
import dockerfile from "@shikijs/langs/dockerfile"
import toml from "@shikijs/langs/toml"
import ini from "@shikijs/langs/ini"
import diff from "@shikijs/langs/diff"
// Shiki ships no LikeC4 grammar: this is the VS Code extension's TextMate grammar,
// vendored unchanged from likec4/likec4@v1.59.4 packages/vscode (MIT, (c) Denis Davydkov),
// pinned to the LikeC4 version src/c4 parses with.
import likec4 from "./grammars/likec4.tmLanguage.json"

let highlighterPromise: Promise<HighlighterCore> | null = null

export function getHighlighter(): Promise<HighlighterCore> {
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

export async function highlightCode(code: string, lang: string): Promise<string> {
  const resolved = LANG_ALIASES[lang] ?? lang
  const hl = await getHighlighter()
  const safeLang = hl.getLoadedLanguages().includes(resolved) ? resolved : "text"
  try {
    return hl.codeToHtml(code, {
      lang: safeLang,
      themes: {
        dark: "andromeeda",
        light: "snazzy-light",
      },
      defaultColor: false,
    })
  } catch {
    try {
      return hl.codeToHtml(code, {
        lang: "text",
        themes: {
          dark: "andromeeda",
          light: "snazzy-light",
        },
        defaultColor: false,
      })
    } catch {
      return ""
    }
  }
}