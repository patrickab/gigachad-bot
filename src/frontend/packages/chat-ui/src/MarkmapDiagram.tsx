"use client"

import { useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, type Ref } from "react"
import type { Markmap } from "markmap-view"
import { CodeBlock, type ColorScheme } from "./CodeBlock.tsx"
import { DIAGRAM_DARK } from "./diagramPalette.ts"
import { fenceCode } from "./selectionToMarkdown.ts"

const MIN_MARKMAP_SCALE = 0.5
const CANVAS_MARKMAP_PADDING = 24

/** The svg d3-zoom drives, with the transform d3 stores on it. */
type D3ZoomSvg = SVGSVGElement & { __zoom?: { k: number; x: number; y: number } }

/** Current d3-zoom transform of the map's svg, if it has one. */
function zoomOf(svg: SVGSVGElement | null) {
  return (svg as D3ZoomSvg | null)?.__zoom
}

/**
 * The map's viewport, for a host that moves its camera (e.g. keeps it centered while an embedding canvas zooms).
 * The map is a viewport pinned to its top-left corner, like React Flow.
 */
export interface MarkmapViewport {
  /** The map's frame, null while the source shows as code. */
  readonly element: HTMLDivElement | null
  /** Pan offset and zoom, null before the map first renders. */
  camera(): { offset: { x: number; y: number }; scale: number } | null
  /** Zoom to `scale` and put the world point `centerX`/`centerY` at the center of a frame of `size`. */
  setCamera(camera: { scale: number; centerX: number; centerY: number }, size: { width: number; height: number }): void
}

/** A markmap fence as a pannable, zoomable mind map in the host's color scheme, the source as a code block when it fails. Styled by `code.css`. */
export function MarkmapDiagram({ code, theme, viewportRef }: { code: string; theme: ColorScheme; viewportRef?: Ref<MarkmapViewport> }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const mmRef = useRef<Markmap | null>(null)
  const [failed, setFailed] = useState(false)
  const [canvasViewportHeight, setCanvasViewportHeight] = useState<number | null>(null)
  const renderedRef = useRef<{ code: string; theme: ColorScheme } | null>(null)

  useImperativeHandle(viewportRef, () => ({
    get element() {
      return containerRef.current
    },
    camera() {
      const transform = zoomOf(svgRef.current)
      return mmRef.current && transform ? { offset: { x: transform.x, y: transform.y }, scale: transform.k } : null
    },
    setCamera({ scale, centerX, centerY }, size) {
      const mm = mmRef.current
      if (!mm) return
      mm.zoom.scaleTo(mm.svg, scale)
      mm.zoom.translateTo(mm.svg, centerX, centerY, [size.width / 2, size.height / 2])
    },
  }), [])

  // A generated document sits under Streamdown's auto-sized <pre>/<code> wrappers.
  // Measure the canvas attachment itself so the map owns its complete viewport, not
  // the wrapper's 400px fallback.
  useLayoutEffect(() => {
    const attachment = containerRef.current?.closest<HTMLElement>("[data-canvas-attachment]")
    if (!attachment) return
    const measure = () => setCanvasViewportHeight(Math.max(0, attachment.clientHeight - CANVAS_MARKMAP_PADDING))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(attachment)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        // Loaded on the first map: markmap and its d3 are large and most messages have none.
        const { Transformer } = await import("markmap-lib")
        const { Markmap } = await import("markmap-view")
        if (cancelled || !containerRef.current) return
        if (containerRef.current.closest("[data-canvas-attachment]") && canvasViewportHeight === null) return
        if (renderedRef.current?.code === code && renderedRef.current.theme === theme) return

        const transformer = new Transformer()
        const { root } = transformer.transform(code)

        if (!svgRef.current) {
          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg")
          svg.style.width = "100%"
          svg.style.height = "100%"
          containerRef.current.appendChild(svg)
          svgRef.current = svg
        }

        const isLight = theme === "light"
        const ink = isLight ? "#1c1917" : DIAGRAM_DARK.ink
        const linkColor = isLight ? "#a8a29e" : DIAGRAM_DARK.inkFaint

        // Markmap's native drag and wheel controls make this a real viewport. Fit only
        // new content. A canvas resize must preserve the user's pan and zoom.
        mmRef.current ??= new Markmap(svgRef.current, {
          autoFit: false,
          duration: 300,
          paddingX: 16,
          initialExpandLevel: -1,
          zoom: true,
          pan: true,
          color: () => linkColor,
        })
        await mmRef.current.setData(root)
        if (renderedRef.current?.code !== code) {
          try {
            await mmRef.current.fit()
            const scale = zoomOf(svgRef.current)?.k ?? 1
            if (scale < MIN_MARKMAP_SCALE) await mmRef.current.rescale(MIN_MARKMAP_SCALE / scale)
          } catch {}
        }
        renderedRef.current = { code, theme }

        // text recolors on theme switch; link color is baked at create (minor)
        svgRef.current.style.setProperty("--markmap-text-color", ink)
      } catch {
        if (!cancelled) setFailed(true)
      }
    })()
    return () => { cancelled = true }
  }, [code, theme, canvasViewportHeight])

  if (failed) return <CodeBlock codeString={code} language="markdown" theme={theme} />
  return <div ref={containerRef} data-markdown-source={fenceCode("markmap", code)} className="gc-markmap" style={{ height: canvasViewportHeight === null ? "var(--markmap-height, 400px)" : `${canvasViewportHeight}px` }} />
}
