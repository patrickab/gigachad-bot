import {
  bundledLanguages,
  bundledLanguagesInfo,
  createHighlighter,
  type BundledLanguage,
  type ThemeRegistrationAny,
  type BundledTheme,
} from "shiki"
import type { CodeHighlighterPlugin, HighlightOptions, HighlightResult } from "streamdown"

type ThemeInput = BundledTheme | ThemeRegistrationAny

const aliasMap = Object.fromEntries(
  bundledLanguagesInfo.flatMap((info) =>
    (info.aliases ?? []).map((alias) => [alias, info.id]),
  ),
) as Record<string, string>

const supported = new Set(Object.keys(bundledLanguages))

function resolveLanguage(language: string): BundledLanguage {
  const normalized = language.trim().toLowerCase()
  const aliased = aliasMap[normalized]
  if (aliased) return aliased as BundledLanguage
  if (supported.has(normalized)) return normalized as BundledLanguage
  return "text"
}

const highlighterCache = new Map<string, ReturnType<typeof createHighlighter>>()
const resultCache = new Map<string, HighlightResult>()
const pendingCallbacks = new Map<string, Set<(result: HighlightResult) => void>>()

function highlighterKey(language: string, themes: [ThemeInput, ThemeInput]) {
  return `${language}-${themes[0]}-${themes[1]}`
}

function resultKey(code: string, language: string, themes: [ThemeInput, ThemeInput]) {
  const head = code.slice(0, 100)
  const tail = code.length > 100 ? code.slice(-100) : ""
  return `${language}:${themes[0]}:${themes[1]}:${code.length}:${head}:${tail}`
}

function getHighlighter(language: string, themes: [ThemeInput, ThemeInput]) {
  const key = highlighterKey(language, themes)
  const existing = highlighterCache.get(key)
  if (existing) return existing
  const created = createHighlighter({ themes, langs: [language] })
  highlighterCache.set(key, created)
  return created
}

export function createCodePlugin(options: { themes?: [ThemeInput, ThemeInput] } = {}): CodeHighlighterPlugin {
  const themes: [ThemeInput, ThemeInput] = options.themes ?? ["github-light", "github-dark"]

  return {
    name: "shiki",
    type: "code-highlighter",
    supportsLanguage(language) {
      return supported.has(resolveLanguage(language))
    },
    getSupportedLanguages() {
      return Array.from(supported) as BundledLanguage[]
    },
    getThemes() {
      return themes
    },
    highlight({ code, language, themes: runtimeThemes }, callback) {
      const resolved = resolveLanguage(language)
      const activeThemes = runtimeThemes ?? themes
      const cacheKey = resultKey(code, resolved, activeThemes)

      const cached = resultCache.get(cacheKey)
      if (cached) return cached

      if (callback) {
        if (!pendingCallbacks.has(cacheKey)) pendingCallbacks.set(cacheKey, new Set())
        pendingCallbacks.get(cacheKey)!.add(callback)
      }

      void getHighlighter(resolved, activeThemes)
        .then(async (highlighter) => {
          const lang = (await highlighter).getLoadedLanguages().includes(resolved) ? resolved : "text"
          const result = (await highlighter).codeToTokens(code, {
            lang,
            themes: { light: activeThemes[0], dark: activeThemes[1] },
          })
          resultCache.set(cacheKey, result)
          const waiters = pendingCallbacks.get(cacheKey)
          if (waiters) {
            for (const notify of waiters) notify(result)
            pendingCallbacks.delete(cacheKey)
          }
        })
        .catch((err) => {
          console.error("[Streamdown Code] Failed to highlight code:", err)
          pendingCallbacks.delete(cacheKey)
        })

      return null
    },
  }
}

export const codePlugin = createCodePlugin()
