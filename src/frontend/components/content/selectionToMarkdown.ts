import TurndownService from "turndown"
import { gfm } from "turndown-plugin-gfm"

/** Elements whose markdown syntax is lost when a selection covers only their text. */
const FORMATTED_ELEMENTS = "h1,h2,h3,h4,h5,h6,li,blockquote,strong,em,del,a,code,[data-code-language],[data-markdown-source]"

/** Fence `code` with a backtick run longer than any run inside it. */
export function fenceCode(language: string, code: string): string {
  const longestRun = Math.max(2, ...(code.match(/`+/g) ?? []).map((run) => run.length))
  const fence = "`".repeat(longestRun + 1)
  return `${fence}${language}\n${code.replace(/\n$/, "")}\n${fence}`
}

function isElement(node: Node): node is HTMLElement {
  return node.nodeType === Node.ELEMENT_NODE
}

function latexOf(formula: HTMLElement): string {
  const annotation = formula.querySelector('annotation[encoding="application/x-tex"]')
  return (annotation ?? formula).textContent ?? ""
}

const turndown = new TurndownService({
  headingStyle: "atx",
  codeBlockStyle: "fenced",
  bulletListMarker: "-",
  emDelimiter: "*",
  hr: "---",
})
turndown.use(gfm)
// Keep the text as written, escaping would turn `snake_case` into `snake\_case`.
turndown.escape = (text) => text
turndown.remove(["button", "style", "script"])
turndown.addRule("katexDisplay", {
  filter: (node) => isElement(node) && node.classList.contains("katex-display"),
  replacement: (_content, node) => `\n\n$$\n${latexOf(node as HTMLElement)}\n$$\n\n`,
})
turndown.addRule("katexInline", {
  filter: (node) => isElement(node) && node.classList.contains("katex"),
  replacement: (_content, node) => `$${latexOf(node as HTMLElement)}$`,
})
// Turndown pads list markers to four columns, use a single space like hand-written markdown.
turndown.addRule("listItem", {
  filter: "li",
  replacement: (content, node, options) => {
    const list = node.parentNode as HTMLElement
    const position = Array.from(list.children).indexOf(node as HTMLElement)
    const marker = list.nodeName === "OL" ? `${(Number(list.getAttribute("start")) || 1) + position}.` : (options.bulletListMarker ?? "-")
    const indent = " ".repeat(marker.length + 1)
    const body = content.replace(/^\n+/, "").replace(/\n+$/, "\n").replace(/\n/g, `\n${indent}`)
    return `${marker} ${body}${node.nextSibling && !body.endsWith("\n") ? "\n" : ""}`
  },
})
turndown.addRule("codeBlock", {
  filter: (node) => isElement(node) && node.hasAttribute("data-code-language"),
  replacement: (_content, node) => {
    const element = node as HTMLElement
    return `\n\n${fenceCode(element.dataset.codeLanguage ?? "", element.querySelector("pre")?.textContent ?? "")}\n\n`
  },
})
turndown.addRule("markdownSource", {
  filter: (node) => isElement(node) && node.hasAttribute("data-markdown-source"),
  replacement: (_content, node) => `\n\n${(node as HTMLElement).dataset.markdownSource}\n\n`,
})

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim()
}

/** Whole formula when the selection touches one, else the outermost element whose text is fully selected. */
function findSourceElement(range: Range, selectedText: string, root: HTMLElement): HTMLElement | null {
  const container = range.commonAncestorContainer
  let element = isElement(container) ? container : container.parentElement
  if (!element || !root.contains(element)) return null

  const formula = element.closest<HTMLElement>(".katex-display") ?? element.closest<HTMLElement>(".katex")
  if (formula && root.contains(formula)) return formula

  let fullySelected: HTMLElement | null = null
  for (; element && element !== root; element = element.parentElement) {
    if (element.matches(FORMATTED_ELEMENTS) && normalizeWhitespace(element.textContent ?? "") === selectedText) {
      fullySelected = element
    }
  }
  return fullySelected
}

/** Clone a list item inside a copy of its list so it keeps its bullet or number. */
function cloneWithListContext(item: HTMLElement): Node {
  const list = item.parentElement
  if (!list || !/^(UL|OL)$/.test(list.tagName)) return item.cloneNode(true)
  const listClone = list.cloneNode(false) as HTMLElement
  if (list.tagName === "OL") {
    const position = Array.from(list.children).indexOf(item)
    listClone.setAttribute("start", String((Number(list.getAttribute("start")) || 1) + position))
  }
  listClone.append(item.cloneNode(true))
  return listClone
}

/** Markdown for the current selection inside `root`, or null when nothing is selected. */
export function selectionToMarkdown(selection: Selection, root: HTMLElement): string | null {
  if (selection.isCollapsed || selection.rangeCount === 0) return null
  const range = selection.getRangeAt(0)
  const source = findSourceElement(range, normalizeWhitespace(selection.toString()), root)

  const wrapper = document.createElement("div")
  if (!source) wrapper.append(range.cloneContents())
  else wrapper.append(source.tagName === "LI" ? cloneWithListContext(source) : source.cloneNode(true))
  return turndown.turndown(wrapper).trim() || null
}
