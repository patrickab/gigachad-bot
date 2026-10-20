import type { ComponentType } from "react"
import type { Components } from "streamdown"

/** Props of the host's fenced-code component. `plain` skips highlighting, for diagrams drawn with text characters. */
export interface CodeBlockProps {
  codeString: string
  language: string
  plain?: boolean
}

/** Host-supplied renderers for fenced code. A host without a diagram renderer shows those fences as code. */
export interface CodeRenderers {
  code: ComponentType<CodeBlockProps>
  mermaid?: ComponentType<{ code: string }>
  markmap?: ComponentType<{ code: string }>
}

// Detect mermaid code in fences without a language tag by its diagram keyword.
function isMermaidCode(code: string): boolean {
  return /^\s*(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph|journey|requirementDiagram|mindmap|timeline|sankey|xychart|block-beta|packet-beta|architecture-beta)\b/i.test(code)
}

// Box-drawing (U+2500-257F) and block element (U+2580-259F) chars: treat as a diagram when dense enough.
function containsAsciiTableArt(code: string): boolean {
  const boxChars = (code.match(/[─-╿▀-▟]/g) || []).length
  const lines = code.split("\n").length
  return boxChars >= 8 && lines >= 3
}

/**
 * Plain elements instead of Streamdown's Tailwind-styled defaults, so the package's `markdown.css` styles them.
 * Fenced code is routed to the host's renderers.
 */
export function proseComponents(renderers: CodeRenderers): Components {
  const { code: Code, mermaid: Mermaid, markmap: Markmap } = renderers
  return {
    h1: "h1",
    h2: "h2",
    h3: "h3",
    h4: "h4",
    h5: "h5",
    h6: "h6",
    p: "p",
    ul: "ul",
    ol: "ol",
    li: "li",
    blockquote: "blockquote",
    hr: "hr",
    strong: "strong",
    em: "em",
    del: "del",
    // eslint-disable-next-line @next/next/no-img-element -- the host's markdown CSS sizes plain images, next/image can't optimize runtime URLs
    img: ({ src, alt, ...props }) => <img alt={alt ?? ""} src={src} {...props} />,
    table: ({ children, ...props }) => (
      <div className="gc-markdown-table-wrap">
        <table {...props}>{children}</table>
      </div>
    ),
    thead: "thead",
    tbody: "tbody",
    tr: "tr",
    th: "th",
    td: "td",
    inlineCode: (props) => <code {...props} />,
    code: ({ className, children }) => {
      const match = /language-(\w+)/.exec(className || "")
      const codeString = String(children).replace(/\n$/, "")
      if (Markmap && (match?.[1] === "markmap" || className?.includes("markmap"))) {
        return <Markmap code={codeString} />
      }
      if (Mermaid && (match?.[1] === "mermaid" || isMermaidCode(codeString))) {
        return <Mermaid code={codeString} />
      }
      if (!match && containsAsciiTableArt(codeString)) {
        return <Code codeString={codeString} language="text" plain />
      }
      return <Code codeString={codeString} language={match?.[1] || "text"} />
    },
  }
}
