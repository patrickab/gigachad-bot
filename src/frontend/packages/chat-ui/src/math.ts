const CODE_SEGMENT = /(<(code|pre|kbd|samp|script|style)\b[\s\S]*?<\/\2>|<[a-zA-Z\/][^>\n]*>|```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`)/g
const isSpace = (c: string | undefined) => c === undefined || /\s/.test(c)

/** Escape the `$` of `text` that are not Pandoc single-dollar math (see `escapeCurrencyDollars`). */
function escapeTextDollars(text: string): string {
  let out = ""
  let i = 0
  while (i < text.length) {
    const c = text[i]
    if (c === "\\") {
      out += text.slice(i, i + 2)
      i += 2
    } else if (c !== "$") {
      out += c
      i++
    } else if (text[i + 1] === "$") {
      // `$$…$$` display math is left to remark-math.
      out += "$$"
      i += 2
    } else {
      const end = singleDollarEnd(text, i)
      if (end === -1) {
        out += "\\$"
        i++
      } else {
        out += text.slice(i, end + 1)
        i = end + 1
      }
    }
  }
  return out
}

/** Index of the closing `$` of the math opened at `open`, or -1 when it is a literal dollar. */
function singleDollarEnd(text: string, open: number): number {
  if (isSpace(text[open + 1])) return -1
  for (let j = open + 1; j < text.length; j++) {
    const c = text[j]
    if (c === "\\") j++
    else if (c === "\n" && text[j + 1] === "\n") return -1
    else if (c === "$" && !isSpace(text[j - 1]) && !/\d/.test(text[j + 1] ?? "")) return j
  }
  return -1
}

/**
 * Pandoc's rule for single-dollar math: an opening `$` is followed by a non-space, a closing `$` is
 * preceded by a non-space and not followed by a digit. Every other `$` becomes `\$`, so "costs $5 and $10"
 * stays text. Code spans, fences and already escaped `\$` are untouched. Run before `normalizeMathDelimiters`.
 */
export function escapeCurrencyDollars(input: string): string {
  if (input.indexOf("$") === -1) return input
  return input
    .split(CODE_SEGMENT)
    .map((seg, i, all) => {
      if (i % 3 === 2) return ""
      if (i % 3 === 1) return seg
      // Text between two HTML tags is raw HTML, which remark does not read as markdown: `htmlMathToCode` owns it.
      const inHtml = i > 2 && i < all.length - 1 && all[i - 2].startsWith("<") && all[i + 1].startsWith("<")
      return inHtml ? seg : escapeTextDollars(seg)
    })
    .join("")
}

/** Rewrite `\( \)` and `\[ \]` to `$` and `$$` outside code spans and fences. */
export function normalizeMathDelimiters(input: string): string {
  if (input.indexOf("\\(") === -1 && input.indexOf("\\[") === -1) return input
  const segments = input.split(/(```[\s\S]*?```|`[^`]*`)/g)
  return segments
    .map((seg, i) => {
      if (i % 2 === 1) return seg
      return seg
        .replace(/\\\[/g, () => "$$")
        .replace(/\\\]/g, () => "$$")
        .replace(/\\\(/g, () => "$")
        .replace(/\\\)/g, () => "$")
    })
    .join("")
}

// `$$tex$$`, or `$tex$` without space inside its dollars and no digit after (so `$5 and $10` stays text).
const DOLLAR_MATH = /\$\$([\s\S]+?)\$\$|\$(?=\S)((?:\\\$|[^$])*?\S)\$(?!\d)/g
const HTML_TAG = /(<[^>]*>)/
const VERBATIM_TAG = /^<(\/?)(code|pre|kbd|samp|script|style)\b/i

/** `html` with its `$`/`$$` math as `language-math` code, which KaTeX renders, outside verbatim tags. */
export function htmlMathToCode(html: string): string {
  if (!html.includes("$")) return html
  let verbatimDepth = 0
  return html
    .split(HTML_TAG)
    .map((part, i) => {
      if (i % 2 === 1) {
        const verbatim = VERBATIM_TAG.exec(part)
        if (verbatim && !part.endsWith("/>")) verbatimDepth = Math.max(0, verbatimDepth + (verbatim[1] ? -1 : 1))
        return part
      }
      if (verbatimDepth > 0) return part
      return part.replace(DOLLAR_MATH, (_, display: string | undefined, inline: string | undefined) =>
        display !== undefined
          ? `<pre><code class="language-math">${display}</code></pre>`
          : `<code class="language-math">${inline}</code>`)
    })
    .join("")
}

interface MarkdownNode {
  type: string
  value?: string
  children?: MarkdownNode[]
}

function rewriteHtmlMath(node: MarkdownNode): void {
  if (node.type === "html" && node.value) node.value = htmlMathToCode(node.value)
  node.children?.forEach(rewriteHtmlMath)
}

/**
 * Remark plugin: render `$…$` math inside raw HTML, which remark-math never sees (MinerU's `<table>` cells).
 *
 * - KaTeX picks the `language-math` code up after the sanitizer, which keeps `language-*` classes
 */
export function remarkHtmlMath() {
  return rewriteHtmlMath
}
