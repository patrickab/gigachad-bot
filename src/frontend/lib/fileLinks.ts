// Local file links in agent answers, after kahme247/ompweb `lib/file-links.ts` (MIT, Copyright (c) 2026 agegr).
import type { Link, Nodes, Root } from "mdast"

const SCHEME = /^[a-zA-Z][a-zA-Z0-9+.-]*:/
const LINE_SUFFIX = /(?::\d+){1,2}$/

/**
 * Absolute path a markdown link or inline code names, null for anything else (URLs, anchors, API paths).
 *
 * - Relative paths resolve against `base`, a trailing `:line[:col]` is dropped, `~/` stays for the file API to expand
 */
export function resolveLocalFileHref(href: string, base: string): string | null {
  let path = decodeURIComponent(href.split("#", 1)[0].split("?", 1)[0].trim())
  if (path.startsWith("file://")) path = path.slice("file://".length)
  else if (!path || SCHEME.test(path) || path.startsWith("//") || path.startsWith("/api/")) return null
  path = path.replace(LINE_SUFFIX, "")
  if (path.startsWith("/") || path.startsWith("~/")) return path
  const segments = `${base.replace(/\/+$/, "")}/${path}`.split("/")
  const resolved: string[] = []
  for (const segment of segments) {
    if (segment === "..") resolved.pop()
    else if (segment !== "." && segment !== "") resolved.push(segment)
  }
  return `/${resolved.join("/")}`
}

const PATH_SEGMENT = String.raw`[\w.@+~\[\]-]+`
/**
 * Inline code that is exactly one local path, with an optional `:line[:col]`. Conservative: spaces, quotes, `?`, `#` or a scheme stay code.
 *
 * - Absolute with 2+ segments (`/var/tmp/x.png`, `/compact` stays a command), `~/` paths
 * - Relative with a directory and a letter-led extension (`src/a.ts`, `./b.md`, while `a/b` and `x/2.5` stay code)
 */
const INLINE_CODE_PATH = new RegExp(
  "^(?:"
    + `/${PATH_SEGMENT}(?:/${PATH_SEGMENT})+`
    + `|~/${PATH_SEGMENT}(?:/${PATH_SEGMENT})*`
    + `|(?:${PATH_SEGMENT}/)+[\\w.@+~\\[\\]-]*\\.[A-Za-z][A-Za-z0-9]*`
    + String.raw`)(?::\d+(?::\d+)?)?$`,
)

/** Remark plugin: inline code holding one local path becomes a link to it, code already inside a link stays. */
export function remarkInlineCodeFileLinks() {
  const linkInlineCode = (node: Nodes): void => {
    if (!("children" in node) || node.type === "link" || node.type === "linkReference") return
    const children = node.children as Nodes[]
    children.forEach((child, index) => {
      // The URL sanitizer reads a colon before any `/` (`b.txt:3`) as a scheme, `resolveLocalFileHref` decodes `%3A`
      if (child.type === "inlineCode" && INLINE_CODE_PATH.test(child.value)) children[index] = { type: "link", url: child.value.replaceAll(":", "%3A"), children: [child] } satisfies Link
      else linkInlineCode(child)
    })
  }
  return (tree: Root) => linkInlineCode(tree)
}
