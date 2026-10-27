"use client"

import { fenceCode } from "@gigachad/chat-ui"
import { useContext, useEffect, useLayoutEffect, useRef, useState } from "react"
import type { Markmap } from "markmap-view"
import { HostScaleContext } from "@/contexts/HostScaleContext"
import { useThemeName } from "@/hooks/useThemeName"
import { useViewportReframe } from "@/hooks/useViewportReframe"
import { offsetToCamera } from "@/lib/domain/viewport"
import type { ThemeName } from "@/lib/palette"
import { CodeBlock } from "../CodeBlock"
import { DIAGRAM_DARK } from "./diagramPalette"

const MIN_MARKMAP_SCALE = 0.5
const CANVAS_MARKMAP_PADDING = 24

/** The svg d3-zoom drives, with the transform d3 stores on it. */
type D3ZoomSvg = SVGSVGElement & { __zoom?: { k: number; x: number; y: number } }

/** Current d3-zoom transform of the map's svg, if it has one. */
function zoomOf(svg: SVGSVGElement | null) {
  return (svg as D3ZoomSvg | null)?.__zoom
}

/** A markmap fence as a pannable, zoomable mind map, the source as a code block when it fails. */
export function MarkmapDiagram({ code }: { code: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const mmRef = useRef<Markmap | null>(null)
  const [failed, setFailed] = useState(false)
  const themeName = useThemeName()
  const [canvasViewportHeight, setCanvasViewportHeight] = useState<number | null>(null)
  const renderedRef = useRef<{ code: string; themeName: ThemeName } | null>(null)

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

  // The map is a viewport pinned to its top-left corner, like React Flow. Keep
  // the world point at its center in the center when the frame resizes, and
  // when it sits in a canvas, magnify it with the canvas's zoom so the frame
  // shows the same view as the canvas zooms, just bigger or smaller.
  const hostScale = useContext(HostScaleContext)
  useViewportReframe(containerRef, {
    hostScale,
    ready: true,
    skipFromEmpty: true,
    read: () => {
      const transform = zoomOf(svgRef.current)
      return mmRef.current && transform ? { offset: { x: transform.x, y: transform.y }, scale: transform.k } : null
    },
    write: (offset, scale, size) => {
      const mm = mmRef.current!
      const camera = offsetToCamera(offset, scale, size)
      mm.zoom.scaleTo(mm.svg, camera.scale)
      mm.zoom.translateTo(mm.svg, camera.centerX, camera.centerY, [size.width / 2, size.height / 2])
    },
  })

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const { Transformer } = await import("markmap-lib")
        const { Markmap } = await import("markmap-view")
        if (cancelled || !containerRef.current) return
        if (containerRef.current.closest("[data-canvas-attachment]") && canvasViewportHeight === null) return
        if (renderedRef.current?.code === code && renderedRef.current.themeName === themeName) return

        const transformer = new Transformer()
        const { root } = transformer.transform(code)

        if (!svgRef.current) {
          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg")
          svg.style.width = "100%"
          svg.style.height = "100%"
          containerRef.current.appendChild(svg)
          svgRef.current = svg
        }

        const isLight = themeName === "light"
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
        renderedRef.current = { code, themeName }

        // text recolors on theme switch; link color is baked at create (minor)
        svgRef.current.style.setProperty("--markmap-text-color", ink)
      } catch {
        if (!cancelled) setFailed(true)
      }
    })()
    return () => { cancelled = true }
  }, [code, themeName, canvasViewportHeight])

  if (failed) return <CodeBlock codeString={code} language="markdown" />
  return <div ref={containerRef} data-markdown-source={fenceCode("markmap", code)} className="my-3 overflow-hidden rounded-md" style={{ height: canvasViewportHeight === null ? "var(--markmap-height, 400px)" : `${canvasViewportHeight}px` }} />
}
