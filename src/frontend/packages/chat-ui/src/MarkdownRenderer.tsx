"use client"

import { Streamdown, defaultRehypePlugins, defaultRemarkPlugins, type Components } from "streamdown"
import { createMathPlugin } from "@streamdown/math"
import remarkBreaks from "remark-breaks"
import { memo, useCallback, useLayoutEffect, useMemo, useRef, type ClipboardEvent as ReactClipboardEvent, type ComponentProps, type CSSProperties } from "react"
import { proseComponents, type CodeRenderers } from "./components.tsx"
import { escapeCurrencyDollars, normalizeMathDelimiters, remarkHtmlMath } from "./math.ts"
import { selectionToMarkdown } from "./selectionToMarkdown.ts"

type StreamdownProps = ComponentProps<typeof Streamdown>
type PluginList = NonNullable<StreamdownProps["remarkPlugins"]>
type RehypePlugin = NonNullable<StreamdownProps["rehypePlugins"]>[number]

const mathPlugin = createMathPlugin({ singleDollarTextMath: true })
const PLUGINS = { math: mathPlugin }
const LINK_SAFETY_OFF = { enabled: false }
const DEFAULT_REMARK_PLUGINS: PluginList = [...Object.values(defaultRemarkPlugins), remarkBreaks, remarkHtmlMath]
/** Streamdown's default rehype chain without its URL hardening. */
const UNHARDENED: RehypePlugin[] = Object.entries(defaultRehypePlugins).flatMap(([name, plugin]) => name === "harden" ? [] : [plugin])

export interface MarkdownRendererProps {
  content: string
  /** Final markdown is rendered block by block while it streams in. */
  streaming?: boolean
  /** Tighter headings, lists, tables and spacing for dense surfaces. */
  compact?: boolean
  style?: CSSProperties
  /** Fenced code, diagrams: the host owns every renderer so the package carries no host styling or backend. */
  codeRenderers: CodeRenderers
  /** Overrides merged over the prose elements, e.g. links, images, tables. Keep the object stable. */
  components?: Components
  /** Remark plugins run after the defaults (GFM, soft line breaks), in order. Keep the array stable. */
  remarkPlugins?: PluginList
  /**
   * Streamdown's URL hardening drops relative paths without a leading `./` and `file:` links.
   * A host that resolves those hrefs itself turns it off. Defaults to on.
   */
  hardenLinks?: boolean
  /** Enables checkbox toggling: receives the markdown with the clicked task item flipped. */
  onContentChange?: (newContent: string) => void
  /** Classes of toggleable checkboxes. */
  checkboxClassName?: string
}

function MarkdownRendererInner({
  content,
  streaming,
  compact,
  style,
  codeRenderers,
  components: hostComponents,
  remarkPlugins: extraRemarkPlugins,
  hardenLinks = true,
  onContentChange,
  checkboxClassName,
}: MarkdownRendererProps) {
  const processed = useMemo(() => normalizeMathDelimiters(escapeCurrencyDollars(content)), [content])

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
  // The latest source, read at toggle time: a handler that changed with every streamed character would rebuild `components`, and Streamdown then re-parses every block.
  const latest = useRef({ content, checkboxLines })
  useLayoutEffect(() => { latest.current = { content, checkboxLines } }, [content, checkboxLines])
  const handleToggle = useCallback((checkbox: HTMLInputElement, wasChecked: boolean) => {
    if (!onContentChange) return
    const root = checkbox.closest("[data-markdown-root]")
    const idx = root ? [...root.querySelectorAll("input[type=checkbox]")].indexOf(checkbox) : -1
    const lineIdx = latest.current.checkboxLines[idx]
    if (lineIdx === undefined) return
    const lines = latest.current.content.split("\n")
    lines[lineIdx] = wasChecked
      ? lines[lineIdx].replace("- [x]", "- [ ]")
      : lines[lineIdx].replace("- [ ]", "- [x]")
    onContentChange(lines.join("\n"))
  }, [onContentChange])

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
    <div data-markdown-root className={rootClass} style={style} onCopy={handleCopy}>
      <Streamdown
        mode={streaming ? "streaming" : "static"}
        isAnimating={false}
        animated={false}
        caret={undefined}
        controls={{ code: false, table: { copy: true, download: true }, mermaid: false }}
        plugins={PLUGINS}
        remarkPlugins={remarkPlugins}
        rehypePlugins={hardenLinks ? undefined : UNHARDENED}
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
