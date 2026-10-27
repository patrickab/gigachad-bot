"use client"

import { useEffect, useEffectEvent, useRef, useState } from "react"
import "@xterm/xterm/css/xterm.css"
import { cn, errorMessage } from "@/lib/utils"
import { neovimSessionUrl } from "@/lib/backend/urls"

const FONT_SIZE = 13
// Local Nerd Fonts first, so icons from the user's config render as in a terminal. "Mono" keeps icons one cell wide.
const NERD_FONTS = "'JetBrainsMono Nerd Font Mono', 'Symbols Nerd Font Mono'"
const POLICY_VIOLATION = 1008
/** Quiet time after the last size change before the terminal refits. */
const FIT_SETTLE_MS = 150
/** Longest wait for Neovim's redraw after a resize before xterm resizes anyway. */
const REDRAW_TIMEOUT_MS = 500
// Synchronized output (DEC mode 2026): xterm holds rendering in between, Neovim wraps each redraw in it.
const SYNC_START = "\x1b[?2026h"
const SYNC_END = "\x1b[?2026l"
// One char per byte, so the ASCII marker can be found in raw PTY output
const BYTE_DECODER = new TextDecoder("latin1")

/** A running terminal whose host element survives React remounts (fullscreen, overlay). */
export interface NeovimTerminal {
  attach: (parent: HTMLElement) => void
  detach: () => void
  fit: () => void
  focus: () => void
}

export type NeovimStatus =
  | { state: "running" }
  | { state: "stopped"; message: string }

export interface NeovimDocument {
  /** Restarts the session when it changes. */
  key: string
  /** App path: absolute for file-root files, so `nvim` starts in their mounted root. */
  path: string
  filename: string
  content: string
  base_content?: string
}

/** Run one backend Neovim session for `document`.
 *
 * - `:w` calls `save(text)`, whose resolved text replaces the buffer if it differs
 * - Yanks (`y`) are copied to the browser clipboard
 * - `:q` with exit code 0 calls `onQuit`
 * - Unmounting or a new `document.key` discards unsaved edits
 */
export function useNeovimTerminal(document: NeovimDocument | null, save: (text: string) => Promise<string>, onQuit: () => void) {
  const [terminal, setTerminal] = useState<NeovimTerminal | null>(null)
  const [status, setStatus] = useState<NeovimStatus>({ state: "running" })
  const [generation, setGeneration] = useState(0)
  const readDocument = useEffectEvent(() => document)
  const saveText = useEffectEvent(save)
  const quit = useEffectEvent(onQuit)

  const documentKey = document?.key ?? null
  useEffect(() => {
    const opened = readDocument()
    if (documentKey === null || !opened) return
    let disposed = false
    let dispose = () => {}

    void (async () => {
      const [{ Terminal }, { FitAddon }, { WebglAddon }] = await Promise.all([import("@xterm/xterm"), import("@xterm/addon-fit"), import("@xterm/addon-webgl")])
      if (disposed) return
      const host = window.document.createElement("div")
      host.className = "h-full w-full font-mono"
      const xterm = new Terminal({ fontSize: FONT_SIZE, cursorBlink: true, allowTransparency: true, theme: { background: "rgba(0, 0, 0, 0)" } })
      const fitAddon = new FitAddon()
      xterm.loadAddon(fitAddon)
      let isOpen = false

      const socket = new WebSocket(neovimSessionUrl())
      socket.binaryType = "arraybuffer"
      let exitCode: number | null = null
      const send = (data: string | Uint8Array<ArrayBuffer>) => { if (socket.readyState === WebSocket.OPEN) socket.send(data) }
      const encoder = new TextEncoder()
      let sentSize = ""
      const sendSize = (rows: number, cols: number) => {
        if (`${rows}x${cols}` === sentSize) return
        sentSize = `${rows}x${cols}`
        send(JSON.stringify({ type: "resize", rows, cols }))
      }

      // A resize swaps in one frame, so the text never flickers: Neovim redraws for the new size first, its
      // output is held, then xterm resizes and parses it under synchronized output, painting only the result.
      // `scanned` keeps the held output's tail, so a sync end split across two messages is still found
      let pendingResize: { rows: number; cols: number; held: Uint8Array[]; scanned: string; timer: ReturnType<typeof setTimeout> } | null = null
      const applyResize = (redrawn: boolean) => {
        const resize = pendingResize
        if (!resize) return
        pendingResize = null
        clearTimeout(resize.timer)
        xterm.write(SYNC_START, () => {
          xterm.resize(resize.cols, resize.rows)
          // The slot may have changed size again meanwhile
          setTimeout(fit)
        })
        resize.held.forEach((chunk) => xterm.write(chunk))
        if (!redrawn) xterm.write(SYNC_END)
      }

      socket.onopen = () => {
        sentSize = `${xterm.rows}x${xterm.cols}`
        send(JSON.stringify({ type: "open", path: opened.path, filename: opened.filename, content: opened.content, base_content: opened.base_content, rows: xterm.rows, cols: xterm.cols }))
      }
      socket.onmessage = async (event: MessageEvent<ArrayBuffer | string>) => {
        if (typeof event.data !== "string") {
          const chunk = new Uint8Array(event.data)
          if (!pendingResize) { xterm.write(chunk); return }
          pendingResize.held.push(chunk)
          const scanned = pendingResize.scanned + BYTE_DECODER.decode(chunk)
          if (scanned.includes(SYNC_END)) applyResize(true)
          else pendingResize.scanned = scanned.slice(1 - SYNC_END.length)
          return
        }
        const message = JSON.parse(event.data)
        if (message.type === "exit") exitCode = message.code
        if (message.type === "yank") { void navigator.clipboard.writeText(message.content).catch(() => {}); return }
        if (message.type !== "save") return
        try {
          send(JSON.stringify({ type: "saved", id: message.id, content: await saveText(message.content) }))
        } catch (error) {
          send(JSON.stringify({ type: "saved", id: message.id, error: errorMessage(error, "Save failed") }))
        }
      }
      socket.onclose = (event) => {
        if (disposed) return
        if (exitCode === 0) { quit(); return }
        setStatus({ state: "stopped", message: stoppedMessage(exitCode, event.code) })
      }
      xterm.onData((data) => send(encoder.encode(data)))
      xterm.onBinary((data) => send(Uint8Array.from(data, (char) => char.charCodeAt(0))))
      xterm.onResize(({ rows, cols }) => sendSize(rows, cols))

      // xterm measures cells on open, so open only once the host is visible. Canvas cannot
      // resolve CSS variables, so the app font is the computed one.
      const fit = () => {
        if (host.clientWidth === 0 || host.clientHeight === 0) return
        if (!isOpen) {
          xterm.options.fontFamily = `${NERD_FONTS}, ${getComputedStyle(host).fontFamily}`
          xterm.open(host)
          isOpen = true
          // WebGL draws box-drawing and block glyphs itself, so borders join like in kitty. Without it the DOM renderer stays.
          try {
            const webgl = new WebglAddon()
            webgl.onContextLoss(() => webgl.dispose())
            xterm.loadAddon(webgl)
          } catch { /* no WebGL2 */ }
          fitAddon.fit()
          return
        }
        const size = fitAddon.proposeDimensions()
        if (!size || Number.isNaN(size.cols) || Number.isNaN(size.rows) || pendingResize) return
        if (size.cols === xterm.cols && size.rows === xterm.rows) return
        if (socket.readyState !== WebSocket.OPEN) { xterm.resize(size.cols, size.rows); return }
        pendingResize = { ...size, held: [], scanned: "", timer: setTimeout(() => applyResize(false), REDRAW_TIMEOUT_MS) }
        sendSize(size.rows, size.cols)
      }

      setStatus({ state: "running" })
      setTerminal({
        attach: (parent) => { parent.appendChild(host); fit() },
        detach: () => host.remove(),
        fit,
        focus: () => xterm.focus(),
      })
      dispose = () => { if (pendingResize) clearTimeout(pendingResize.timer); socket.close(); xterm.dispose() }
    })()

    return () => {
      disposed = true
      dispose()
      setTerminal(null)
    }
  }, [documentKey, generation])

  return { terminal, status, restart: () => setGeneration((value) => value + 1) }
}

function stoppedMessage(exitCode: number | null, closeCode: number): string {
  if (exitCode !== null) return `Neovim exited with code ${exitCode}.`
  if (closeCode === POLICY_VIOLATION) return "Neovim is not enabled for this account."
  return "Lost the connection to Neovim. Unsaved changes are gone."
}

/** Show `terminal` here. Keys stop at this element, so app shortcuts never steal them. */
export function NeovimEditor({ terminal, status, onRestart, className }: {
  terminal: NeovimTerminal | null
  status: NeovimStatus
  onRestart: () => void
  className?: string
}) {
  const slotRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const slot = slotRef.current
    if (!slot) return
    const keepKey = (event: KeyboardEvent) => event.stopPropagation()
    slot.addEventListener("keydown", keepKey)
    return () => slot.removeEventListener("keydown", keepKey)
  }, [])

  useEffect(() => {
    const slot = slotRef.current
    if (!slot || !terminal) return
    terminal.attach(slot)
    // Fit once the size settles: a resize redraws all of Neovim, so fitting during a drag or panel animation flickers.
    // Meanwhile the terminal keeps its size, clipped or padded by the slot.
    let fitTimer: ReturnType<typeof setTimeout> | undefined
    const observer = new ResizeObserver(() => {
      clearTimeout(fitTimer)
      fitTimer = setTimeout(() => terminal.fit(), FIT_SETTLE_MS)
    })
    observer.observe(slot)
    terminal.focus()
    return () => {
      clearTimeout(fitTimer)
      observer.disconnect()
      terminal.detach()
    }
  }, [terminal])

  return (
    <div ref={slotRef} data-neovim-terminal className={cn("relative min-h-0 flex-1 overflow-hidden px-2 pt-1", className)}>
      {status.state === "stopped" && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-paper text-xs text-ink-faint">
          <span>{status.message}</span>
          <button onClick={onRestart} className="rounded px-2 py-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
            Restart Neovim
          </button>
        </div>
      )}
    </div>
  )
}
