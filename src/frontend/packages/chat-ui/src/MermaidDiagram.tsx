"use client"

import { useEffect, useState } from "react"
import { CodeBlock, type ColorScheme } from "./CodeBlock.tsx"
import { DIAGRAM_DARK, DIAGRAM_LIGHT } from "./diagramPalette.ts"
import { fenceCode } from "./selectionToMarkdown.ts"

// ponytail: mutex serializes mermaid renders, it's a singleton that corrupts on concurrent calls
let mermaidQueue = Promise.resolve()

function enqueueMermaidRender(id: string, code: string, theme: ColorScheme): Promise<string> {
  const job = mermaidQueue.then(async () => {
    // Loaded on the first diagram: mermaid is large and most messages have none.
    const mermaid = (await import("mermaid")).default
    const c = theme === "light" ? DIAGRAM_LIGHT : DIAGRAM_DARK

    // theme: "base" is the only mermaid theme fully driven by themeVariables; named
    // themes recompute their own palette and ignore overrides. Re-initialize every
    // render (cheap, inside the mutex) so theme switches never inherit stale config.
    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      securityLevel: "strict",
      fontFamily: "var(--font-sans), system-ui, sans-serif",
      logLevel: "fatal",
      themeVariables: {
        primaryColor: c.surfaceElevated,
        primaryTextColor: c.ink,
        primaryBorderColor: c.inkFaint,
        secondaryColor: c.surface,
        secondaryTextColor: c.inkMuted,
        secondaryBorderColor: c.inkFaint,
        tertiaryColor: c.paper,
        tertiaryTextColor: c.inkSubtle,
        tertiaryBorderColor: c.inkFaint,
        lineColor: c.inkFaint,
        textColor: c.ink,
        mainBkg: c.surfaceElevated,
        nodeBorder: c.inkFaint,
        clusterBkg: c.surface,
        clusterBorder: c.inkFaint,
        edgeLabelBackground: c.paper,
        fontSize: "14px",
      },
    })
    await mermaid.parse(code)
    const { svg } = await mermaid.render(id, code)
    return svg
  })
  mermaidQueue = job.then(() => undefined, () => undefined)
  return job
}

/** A mermaid fence rendered to SVG in the host's color scheme, the source as a code block when it fails. Styled by `code.css`. */
export function MermaidDiagram({ code, theme }: { code: string; theme: ColorScheme }) {
  const key = `${theme}:${code}`
  // svg null: the render failed
  const [result, setResult] = useState<{ key: string; svg: string | null } | null>(null)

  useEffect(() => {
    let cancelled = false
    // Fresh id per render. Reusing one collides with the previous SVG still in the DOM
    // (re-render on theme switch), which makes mermaid.render fail.
    const id = `mmd-${Math.random().toString(36).slice(2)}`
    enqueueMermaidRender(id, code, theme)
      .then((rendered) => { if (!cancelled) setResult({ key, svg: rendered }) })
      .catch(() => { if (!cancelled) setResult({ key, svg: null }) })
    return () => { cancelled = true }
  }, [code, theme, key])

  // Results from a previous code/theme are stale and show as pending.
  const current = result?.key === key ? result : null
  if (current?.svg === null) return <CodeBlock codeString={code} language="mermaid" theme={theme} />
  if (!current?.svg) return <div className="gc-diagram-pending">Rendering diagram…</div>

  return (
    <div
      data-markdown-source={fenceCode("mermaid", code)}
      className="gc-diagram"
      dangerouslySetInnerHTML={{ __html: current.svg }}
    />
  )
}
