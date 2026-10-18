"use client"

import { useEffect, useRef, useState } from "react"
import "@xterm/xterm/css/xterm.css"
import { cn } from "@/lib/utils"
import { neovimSessionUrl } from "@/lib/backend/urls"

const FONT_SIZE = 13
// Local Nerd Fonts first, so icons from the user's config render as in a terminal. "Mono" keeps icons one cell wide.
const NERD_FONTS = "'JetBrainsMono Nerd Font Mono', 'Symbols Nerd Font Mono'"
const POLICY_VIOLATION = 1008

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
}

/** Run one backend Neovim session for `document`.
 *
 * - `:w` calls `save(text)`, whose resolved text replaces the buffer if it differs
 * - `:q` with exit code 0 calls `onQuit`
 * - Unmounting or a new `document.key` discards unsaved edits
 */
export function useNeovimTerminal(document: NeovimDocument | null, save: (text: string) => Promise<string>, onQuit: () => void) {
  const [terminal, setTerminal] = useState<NeovimTerminal | null>(null)
  const [status, setStatus] = useState<NeovimStatus>({ state: "running" })
  const [generation, setGeneration] = useState(0)
  const documentRef = useRef(document)
  const saveRef = useRef(save)
  const onQuitRef = useRef(onQuit)
  useEffect(() => {
    documentRef.current = document
    saveRef.current = save
    onQuitRef.current = onQuit
  })

  const documentKey = document?.key ?? null
  useEffect(() => {
    const opened = documentRef.current
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

      socket.onopen = () => {
        send(JSON.stringify({ type: "open", path: opened.path, filename: opened.filename, content: opened.content, rows: xterm.rows, cols: xterm.cols }))
      }
      socket.onmessage = async (event: MessageEvent<ArrayBuffer | string>) => {
        if (typeof event.data !== "string") { xterm.write(new Uint8Array(event.data)); return }
        const message = JSON.parse(event.data)
        if (message.type === "exit") exitCode = message.code
        if (message.type !== "save") return
        try {
          send(JSON.stringify({ type: "saved", id: message.id, content: await saveRef.current(message.content) }))
        } catch (error) {
          send(JSON.stringify({ type: "saved", id: message.id, error: error instanceof Error ? error.message : "Save failed" }))
        }
      }
      socket.onclose = (event) => {
        if (disposed) return
        if (exitCode === 0) { onQuitRef.current(); return }
        setStatus({ state: "stopped", message: stoppedMessage(exitCode, event.code) })
      }
      xterm.onData((data) => send(encoder.encode(data)))
      xterm.onBinary((data) => send(Uint8Array.from(data, (char) => char.charCodeAt(0))))
      xterm.onResize(({ rows, cols }) => send(JSON.stringify({ type: "resize", rows, cols })))

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
        }
        fitAddon.fit()
      }

      setStatus({ state: "running" })
      setTerminal({
        attach: (parent) => { parent.appendChild(host); fit() },
        detach: () => host.remove(),
        fit,
        focus: () => xterm.focus(),
      })
      dispose = () => { socket.close(); xterm.dispose() }
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
    const observer = new ResizeObserver(() => terminal.fit())
    observer.observe(slot)
    terminal.focus()
    return () => {
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
