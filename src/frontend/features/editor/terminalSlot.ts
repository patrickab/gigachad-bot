"use client"

import { useEffect, useRef, type RefObject } from "react"
import type { Terminal } from "@xterm/xterm"
import type { WebglAddon } from "@xterm/addon-webgl"
import { useEventListener } from "@/hooks/useEventListener"
import { useThemeName } from "@/hooks/useThemeName"
import { NERD_FONTS } from "./terminalTheme"

/** WebSocket close code of a session the account may not open. */
export const POLICY_VIOLATION = 1008

/** A running terminal whose host element survives React remounts. */
export interface TerminalHandle {
  attach: (parent: HTMLElement) => void
  detach: () => void
  fit: () => void
  focus: () => void
  /** Repaint with the colors of the current app theme. */
  restyle: () => void
}

/**
 * Open `xterm` in `host` once the host is visible: "hidden" while it is not, "opened" on the call that opened it, "open" after.
 *
 * xterm measures cells on open. Canvas cannot resolve CSS variables, so the app font is the computed one.
 */
export function openWhenVisible(xterm: Terminal, host: HTMLElement, Webgl: typeof WebglAddon): "hidden" | "opened" | "open" {
  if (host.clientWidth === 0 || host.clientHeight === 0) return "hidden"
  if (xterm.element) return "open"
  xterm.options.fontFamily = `${NERD_FONTS}, ${getComputedStyle(host).fontFamily}`
  xterm.open(host)
  // WebGL draws box-drawing and block glyphs itself, so borders join like in kitty. Without it the DOM renderer stays.
  try {
    const webgl = new Webgl()
    webgl.onContextLoss(() => webgl.dispose())
    xterm.loadAddon(webgl)
  } catch { /* no WebGL2 */ }
  return "opened"
}

/**
 * Show `terminal` in the returned slot: attached, focused, refit on resize and restyled with the theme. Keys stop at the slot, so app shortcuts never steal them.
 *
 * With `fitDelayMs` it refits once the size has been quiet that long, else at once.
 */
export function useTerminalSlot(terminal: TerminalHandle | null, fitDelayMs?: number): RefObject<HTMLDivElement | null> {
  const slotRef = useRef<HTMLDivElement>(null)
  const theme = useThemeName()

  useEventListener(slotRef, "keydown", (event) => event.stopPropagation())

  useEffect(() => {
    const slot = slotRef.current
    if (!slot || !terminal) return
    terminal.attach(slot)
    let fitTimer: ReturnType<typeof setTimeout> | undefined
    const observer = new ResizeObserver(() => {
      if (fitDelayMs === undefined) { terminal.fit(); return }
      clearTimeout(fitTimer)
      fitTimer = setTimeout(() => terminal.fit(), fitDelayMs)
    })
    observer.observe(slot)
    terminal.focus()
    return () => {
      clearTimeout(fitTimer)
      observer.disconnect()
      terminal.detach()
    }
  }, [terminal, fitDelayMs])

  useEffect(() => { terminal?.restyle() }, [terminal, theme])

  return slotRef
}
