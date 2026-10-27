"use client"

import { useEffect, useLayoutEffect, useRef } from "react"
import { useTabActive } from "@/contexts/TabContext"
import { isNeovimKeyEvent } from "@/lib/utils"

type EscapeLayer = { current: () => void }

/** Active layers, innermost (most recently activated) last. */
const layers: EscapeLayer[] = []

// Neovim owns Escape while its terminal has focus, so `:q` is the way out of a window holding it. An IME owns it mid-composition.
function handleEscape(event: KeyboardEvent) {
  const top = layers.at(-1)
  if (event.key !== "Escape" || !top || event.isComposing || isNeovimKeyEvent(event)) return
  event.preventDefault()
  event.stopImmediatePropagation()
  top.current()
}

/**
 * Give Escape to `onEscape` while `active` and its tab is the active one: the most recently activated layer wins, and nothing underneath sees the key.
 *
 * - A modal registers when it opens, an inline form inside it registers when editing starts, so the form closes first
 * - Window capture phase, so document and React handlers below an open layer never fire
 * - Layers in a hidden tab unregister, so Escape reaches the visible tab only
 */
export function useEscapeLayer(onEscape: () => void, active = true) {
  const tabActive = useTabActive()
  const registered = active && tabActive
  const layer = useRef(onEscape)
  useLayoutEffect(() => {
    layer.current = onEscape
  })
  useEffect(() => {
    if (!registered) return
    if (layers.length === 0) window.addEventListener("keydown", handleEscape, true)
    layers.push(layer)
    return () => {
      layers.splice(layers.indexOf(layer), 1)
      if (layers.length === 0) window.removeEventListener("keydown", handleEscape, true)
    }
  }, [registered])
}
