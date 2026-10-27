"use client"

import { fenceCode } from "@gigachad/chat-ui"
import { useEffect, useState } from "react"
import { useThemeName } from "@/hooks/useThemeName"
import { CodeBlock } from "../CodeBlock"
import { DIAGRAM_DARK, DIAGRAM_LIGHT } from "./diagramPalette"

// ponytail: mutex serializes mermaid renders, it's a singleton that corrupts on concurrent calls
let mermaidQueue = Promise.resolve()

function enqueueMermaidRender(id: string, code: string): Promise<string> {
  const job = mermaidQueue.then(async () => {
    const mermaid = (await import("mermaid")).default
    const isLight =
      typeof document !== "undefined" && document.documentElement.classList.contains("light")
    const c = isLight ? DIAGRAM_LIGHT : DIAGRAM_DARK

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

/** A mermaid fence rendered to SVG in the current theme, the source as a code block when it fails. */
export function MermaidDiagram({ code }: { code: string }) {
  const themeName = useThemeName()
  const key = `${themeName}:${code}`
  // svg null: the render failed
  const [result, setResult] = useState<{ key: string; svg: string | null } | null>(null)

  useEffect(() => {
    let cancelled = false
    // Fresh id per render. Reusing one collides with the previous SVG still in the DOM
    // (re-render on theme switch), which makes mermaid.render fail.
    const id = `mmd-${Math.random().toString(36).slice(2)}`
    enqueueMermaidRender(id, code)
      .then((rendered) => { if (!cancelled) setResult({ key, svg: rendered }) })
      .catch(() => { if (!cancelled) setResult({ key, svg: null }) })
    return () => { cancelled = true }
  }, [code, key])

  // Results from a previous code/theme are stale and show as pending.
  const current = result?.key === key ? result : null
  if (current?.svg === null) return <CodeBlock codeString={code} language="mermaid" />
  if (!current?.svg) return <div className="my-3 text-xs text-ink-subtle">Rendering diagram…</div>

  return (
    <div
      data-markdown-source={fenceCode("mermaid", code)}
      className="my-3 flex justify-center overflow-x-auto [&_svg]:max-w-full [&_svg]:h-auto"
      dangerouslySetInnerHTML={{ __html: current.svg }}
    />
  )
}
