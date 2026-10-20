"use client"

import { Streamdown, defaultRehypePlugins, defaultRemarkPlugins, type Components } from "streamdown"
import { createMathPlugin } from "@streamdown/math"
import remarkBreaks from "remark-breaks"
import { memo, useCallback, useMemo, type ClipboardEvent as ReactClipboardEvent, type ComponentProps, type CSSProperties } from "react"
import { proseComponents, type CodeRenderers } from "./components.tsx"
import { normalizeMathDelimiters } from "./math.ts"
import { selectionToMarkdown } from "./selectionToMarkdown.ts"

type StreamdownProps = ComponentProps<typeof Streamdown>
type PluginList = NonNullable<StreamdownProps["remarkPlugins"]>
type RehypePlugin = NonNullable<StreamdownProps["rehypePlugins"]>[number]

const mathPlugin = createMathPlugin({ singleDollarTextMath: true })
const PLUGINS = { math: mathPlugin }
const LINK_SAFETY_OFF = { enabled: false }
const DEFAULT_REMARK_PLUGINS: PluginList = [...Object.values(defaultRemarkPlugins), remarkBreaks]

/** How the host's hrefs pass Streamdown's sanitizer. Omit it to keep Streamdown's defaults. */
export interface LinkPolicy {
  /** Href protocols kept in addition to the sanitizer's own, e.g. `agent` for in-app handles. */
  extraHrefProtocols?: readonly string[]
  /**
   * Streamdown's URL hardening drops relative paths without a leading `./` and `file:` links.
   * A host that resolves those hrefs itself turns it off. Defaults to on.
   */
  harden?: boolean
}

export interface MarkdownRendererProps {
  content: string
  /** Final markdown is rendered block by block while it streams in. */
  streaming?: boolean
  /** Tighter headings, lists, tables and spacing for dense surfaces. Selects the root class, so it is not inferred from `className`. */
  compact?: boolean
  /** Extra classes of the root element, which always carries `data-markdown-root` and its own `gc-markdown` / `gc-markdown-compact` class. */
  className?: string
  style?: CSSProperties
  /** Fenced code, diagrams: the host owns every renderer so the package carries no host styling or backend. */
  codeRenderers: CodeRenderers
  /** Overrides merged over the prose elements, e.g. links, images, tables. Keep the object stable. */
  components?: Components
  /** Remark plugins run after the defaults (GFM, soft line breaks), in order. Keep the array stable. */
  remarkPlugins?: PluginList
  linkPolicy?: LinkPolicy
  /** Enables checkbox toggling: receives the markdown with the clicked task item flipped. */
  onContentChange?: (newContent: string) => void
  /** Classes of toggleable checkboxes. */
  checkboxClassName?: string
}

/** Streamdown's default rehype chain, with the link policy applied to its sanitizer and hardener. */
function rehypePluginsFor(extraHrefProtocols: readonly string[], harden: boolean): RehypePlugin[] {
  return Object.entries(defaultRehypePlugins).flatMap(([name, plugin]): RehypePlugin[] => {
    if (name === "harden" && !harden) return []
    if (name !== "sanitize" || extraHrefProtocols.length === 0) return [plugin]
    if (!Array.isArray(plugin) || typeof plugin[1] !== "object" || plugin[1] === null) {
      throw new Error("streamdown changed the shape of its sanitize plugin, update rehypePluginsFor")
    }
    const [sanitize, schema] = plugin as [unknown, { protocols?: { href?: readonly string[] } }]
    const protocols = { ...schema.protocols, href: [...(schema.protocols?.href ?? []), ...extraHrefProtocols] }
    return [[sanitize, { ...schema, protocols }] as RehypePlugin]
  })
}

function MarkdownRendererInner({
  content,
  streaming,
  compact,
  className,
  style,
  codeRenderers,
  components: hostComponents,
  remarkPlugins: extraRemarkPlugins,
  linkPolicy,
  onContentChange,
  checkboxClassName,
}: MarkdownRendererProps) {
  const processed = useMemo(() => normalizeMathDelimiters(content), [content])

  // Source line index of each task item, only tracked when toggling is enabled.
  const togglingEnabled = !!onContentChange
  const checkboxLines = useMemo(() => {
    if (!togglingEnabled) return []
    const lines = content.split("\n")
    const indices: number[] = []
    for (let i = 0; i < lines.length; i++) {
      if (/^- \[[ x]\]/.test(lines[i].trimStart())) indices.push(i)
    }
    return indices
  }, [content, togglingEnabled])

  // Rendered checkboxes follow source order, so a checkbox's index among its root's checkboxes is its task index.
  // Read from the DOM at toggle time: Streamdown re-renders blocks independently, so render order can't number them.
  const handleToggle = useCallback((checkbox: HTMLInputElement, wasChecked: boolean) => {
    if (!onContentChange) return
    const root = checkbox.closest("[data-markdown-root]")
    const idx = root ? [...root.querySelectorAll("input[type=checkbox]")].indexOf(checkbox) : -1
    const lineIdx = checkboxLines[idx]
    if (lineIdx === undefined) return
    const lines = content.split("\n")
    lines[lineIdx] = wasChecked
      ? lines[lineIdx].replace("- [x]", "- [ ]")
      : lines[lineIdx].replace("- [ ]", "- [x]")
    onContentChange(lines.join("\n"))
  }, [content, checkboxLines, onContentChange])

  const components = useMemo<Components>(() => {
    const base: Components = { ...proseComponents(codeRenderers), ...hostComponents }
    if (onContentChange) {
      base.input = ({ checked, type, ...rest }) => {
        if (type !== "checkbox") return <input type={type} checked={checked} {...rest} />
        return (
          <input
            type="checkbox"
            checked={checked}
            className={checkboxClassName}
            onChange={(event) => handleToggle(event.currentTarget, !!checked)}
          />
        )
      }
    }
    return base
  }, [codeRenderers, hostComponents, onContentChange, handleToggle, checkboxClassName])

  const remarkPlugins = useMemo<PluginList>(
    () => (extraRemarkPlugins?.length ? [...DEFAULT_REMARK_PLUGINS, ...extraRemarkPlugins] : DEFAULT_REMARK_PLUGINS),
    [extraRemarkPlugins],
  )
  // The policy is plain data: compare it by value, so an inline object does not rebuild the chain.
  const hasLinkPolicy = linkPolicy !== undefined
  const harden = linkPolicy?.harden !== false
  const extraProtocols = linkPolicy?.extraHrefProtocols?.join(" ") ?? ""
  const rehypePlugins = useMemo(
    () => (hasLinkPolicy ? rehypePluginsFor(extraProtocols ? extraProtocols.split(" ") : [], harden) : undefined),
    [hasLinkPolicy, extraProtocols, harden],
  )

  // Copy the underlying markdown (headings, math, code fences) instead of the rendered text, the exact source when all of it is selected.
  const handleCopy = useCallback((event: ReactClipboardEvent<HTMLDivElement>) => {
    const selection = window.getSelection()
    if (!selection) return
    const renderedText = event.currentTarget.innerText ?? event.currentTarget.textContent ?? ""
    const selectsAll = selection.toString().trim() === renderedText.trim()
    const markdown = selectsAll ? content : selectionToMarkdown(selection, event.currentTarget)
    if (!markdown) return
    event.preventDefault()
    event.clipboardData.setData("text/plain", markdown)
  }, [content])

  const rootClass = compact ? "gc-markdown-compact" : "gc-markdown"

  return (
    <div data-markdown-root className={className ? `${rootClass} ${className}` : rootClass} style={style} onCopy={handleCopy}>
      <Streamdown
        mode={streaming ? "streaming" : "static"}
        isAnimating={false}
        animated={false}
        caret={undefined}
        controls={{ code: false, table: { copy: true, download: true }, mermaid: false }}
        plugins={PLUGINS}
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        linkSafety={LINK_SAFETY_OFF}
        components={components}
      >
        {processed}
      </Streamdown>
    </div>
  )
}

/**
 * Markdown with KaTeX math, host-rendered code and diagrams, copy-as-markdown and optional checkbox toggling.
 * The host loads `katex/dist/katex.min.css` and the package's `markdown.css`, a CSS import in the package would break server-side externalization.
 */
export const MarkdownRenderer = memo(MarkdownRendererInner)
