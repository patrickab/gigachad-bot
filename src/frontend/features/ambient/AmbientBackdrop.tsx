"use client"

import { useEffect, useRef, useState } from "react"
import { readSceneAnchor, startNightOcean, type NightOceanFrame } from "deepfocus-ambient"
import { useThemeName } from "@/hooks/useThemeName"
import { readToken } from "@/lib/palette"

/** The baked sea, fetched by `npm run ambient:fetch`. Without the file the sea is computed live. */
const VIDEO_URL = "/ambient/night-ocean.webm"
/** The scene before any anchor was on screen: about where an empty conversation puts it. */
const DEFAULT_SCENE = { horizon: 0.4, moonX: 0.5, moonY: 0.48 }
/** Share of the way to a moved anchor covered per frame, so the scene glides instead of jumping. */
const FOLLOW = 0.16
/** A graphics context the browser took away and did not give back by then is replaced, with a fresh canvas. */
const CONTEXT_WAIT_MS = 3000

function toRgb(cssColor: string): [number, number, number] {
  const context = document.createElement("canvas").getContext("2d")
  if (!context) return [0, 0, 0]
  context.fillStyle = cssColor
  context.fillRect(0, 0, 1, 1)
  const [r, g, b] = context.getImageData(0, 0, 1, 1).data
  return [r / 255, g / 255, b / 255]
}

/**
 * The night ocean of `deepfocus-ambient` behind the whole app, in the theme's colors.
 *
 * - Drawn in `--ink` and `--paper`: by night in the dark theme, by day in the light theme (a white sun in haze over a darker sea)
 * - One fixed canvas under everything: it shows wherever the shell paints no surface of its own
 * - The scene follows the newest `anchorScene` on screen (an empty conversation's composer) and stays where that left it
 * - Reduced motion gets a still sea
 * - The canvas is hidden while it has no picture (before the first frame, after a lost graphics context), the page color shows. A context that stays lost gets a fresh canvas
 */
export function AmbientBackdrop() {
  const theme = useThemeName()
  const [hasFrame, setHasFrame] = useState(false)
  /** Counts the canvases used up: each lost graphics context that stayed lost. */
  const [canvasKey, setCanvasKey] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const colorsRef = useRef<Pick<NightOceanFrame, "paper" | "light">>({ paper: [0, 0, 0], light: [1, 1, 1] })

  // Before the effect below, so its first frame has the colors
  useEffect(() => {
    colorsRef.current = { paper: toRgb(readToken("paper")), light: toRgb(readToken("ink")) }
  }, [theme])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const still = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false
    let scene: typeof DEFAULT_SCENE | null = null
    const frame = (): NightOceanFrame => {
      const anchor = readSceneAnchor()
      const { clientWidth: width, clientHeight: height } = canvas
      if (anchor && width > 0 && height > 0) {
        const target = { horizon: 1 - anchor.horizonY / height, moonX: anchor.lightX / width, moonY: 1 - anchor.lightY / height }
        const follow = scene && !still ? FOLLOW : 1
        const from = scene ?? target
        scene = {
          horizon: from.horizon + (target.horizon - from.horizon) * follow,
          moonX: from.moonX + (target.moonX - from.moonX) * follow,
          moonY: from.moonY + (target.moonY - from.moonY) * follow,
        }
      }
      const { horizon, moonX, moonY } = scene ?? DEFAULT_SCENE
      return { ...colorsRef.current, horizon, moon: [moonX, moonY], still }
    }
    let replace: ReturnType<typeof setTimeout> | undefined
    const stop = startNightOcean(canvas, {
      frame,
      videoUrl: VIDEO_URL,
      onFirstFrame: () => {
        clearTimeout(replace)
        setHasFrame(true)
      },
      onLost: () => {
        setHasFrame(false)
        replace = setTimeout(() => setCanvasKey((key) => key + 1), CONTEXT_WAIT_MS)
      },
    })
    return () => {
      clearTimeout(replace)
      stop?.()
      setHasFrame(false)
    }
  }, [canvasKey])

  return (
    <canvas
      key={canvasKey}
      ref={canvasRef}
      aria-hidden="true"
      className={`pointer-events-none fixed inset-0 -z-10 h-full w-full transition-opacity duration-1000 ease-out motion-reduce:transition-none ${hasFrame ? "opacity-100" : "opacity-0"}`}
    />
  )
}
