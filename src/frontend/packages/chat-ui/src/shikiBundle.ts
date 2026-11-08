/** Shiki with every grammar and theme the code blocks use: one chunk `syntaxHighlighting` loads on the first highlight, kept out of the page bundle. */
import type { LanguageRegistration } from "shiki/core"
export { createHighlighterCore } from "shiki/core"
export { createJavaScriptRegexEngine } from "shiki/engine/javascript"
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

export const themes = [andromeeda, snazzyLight]

export const langs = [
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
]
