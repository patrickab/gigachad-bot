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
