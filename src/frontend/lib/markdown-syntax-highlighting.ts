import { createHighlighterCoreSync, type HighlighterCore } from "shiki/core"
import { createJavaScriptRegexEngine } from "shiki/engine/javascript"
import darkPlus from "@shikijs/themes/dark-plus"
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

let highlighter: HighlighterCore | null = null

export function getHighlighter(): HighlighterCore {
  if (!highlighter) {
    highlighter = createHighlighterCoreSync({
      themes: [darkPlus],
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
      ],
      engine: createJavaScriptRegexEngine(),
    })
  }
  return highlighter
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
}

export function highlightCode(code: string, lang: string): string {
  const resolved = LANG_ALIASES[lang] ?? lang
  const hl = getHighlighter()
  const safeLang = hl.getLoadedLanguages().includes(resolved) ? resolved : "text"
  try {
    return hl.codeToHtml(code, { lang: safeLang, theme: "dark-plus" })
  } catch {
    try {
      return hl.codeToHtml(code, { lang: "text", theme: "dark-plus" })
    } catch {
      return ""
    }
  }
}