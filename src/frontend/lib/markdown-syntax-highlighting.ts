import { createHighlighterCore, type HighlighterCore } from "shiki/core"
import { createJavaScriptRegexEngine } from "shiki/engine/javascript"
import darkPlus from "@shikijs/themes/dark-plus"
import lightPlus from "@shikijs/themes/light-plus"
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

let highlighterPromise: Promise<HighlighterCore> | null = null

export function getHighlighter(): Promise<HighlighterCore> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighterCore({
      themes: [darkPlus, lightPlus],
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
}

export async function highlightCode(code: string, lang: string): Promise<string> {
  const resolved = LANG_ALIASES[lang] ?? lang
  const hl = await getHighlighter()
  const safeLang = hl.getLoadedLanguages().includes(resolved) ? resolved : "text"
  try {
    return hl.codeToHtml(code, {
      lang: safeLang,
      themes: {
        dark: "dark-plus",
        light: "light-plus",
      },
      defaultColor: false,
    })
  } catch {
    try {
      return hl.codeToHtml(code, {
        lang: "text",
        themes: {
          dark: "dark-plus",
          light: "light-plus",
        },
        defaultColor: false,
      })
    } catch {
      return ""
    }
  }
}