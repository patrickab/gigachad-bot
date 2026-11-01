"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react"
import "@xterm/xterm/css/xterm.css"
import { cn, errorMessage } from "@/lib/utils"
import { neovimSessionUrl } from "@/lib/backend/urls"
import { useEventListener } from "@/hooks/useEventListener"
import { useThemeName } from "@/hooks/useThemeName"
import { readToken } from "@/lib/palette"

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

type Rgb = [red: number, green: number, blue: number]

/** `color` as RGB, null when fully transparent. Canvas resolves any CSS color, xterm only takes opaque ones. */
function toRgb(color: string): Rgb | null {
  const context = window.document.createElement("canvas").getContext("2d", { willReadFrequently: true })
  if (!context) return null
  context.fillStyle = color
  context.fillRect(0, 0, 1, 1)
  const [red, green, blue, alpha] = context.getImageData(0, 0, 1, 1).data
  return alpha === 0 ? null : [red, green, blue]
}

const rgbCss = ([red, green, blue]: Rgb) => `rgb(${red}, ${green}, ${blue})`

/** Share of ink in the cursor block over the background: clearly visible, softer than full ink. */
const CURSOR_INK_SHARE = 0.3

const mix = (base: Rgb, over: Rgb, share: number): Rgb => base.map((channel, index) => Math.round(channel + (over[index] - channel) * share)) as Rgb

/** Terminal colors of the app theme: the background `element` sits on, and the ink on it. `isLight` is what Neovim gets as 'background'. */
function terminalTheme(element: HTMLElement) {
  let background: Rgb | null = null
  for (let node: HTMLElement | null = element; node && !background; node = node.parentElement) background = toRgb(getComputedStyle(node).backgroundColor)
  background ??= toRgb(readToken("paper")) ?? [0, 0, 0]
  const ink = toRgb(readToken("ink")) ?? [255, 255, 255]
  const [red, green, blue] = background
  return {
    // The character under the cursor block keeps the ink
    theme: { background: rgbCss(background), foreground: rgbCss(ink), cursor: rgbCss(mix(background, ink, CURSOR_INK_SHARE)), cursorAccent: rgbCss(ink) },
    isLight: 0.299 * red + 0.587 * green + 0.114 * blue > 128,
  }
}

/** A running terminal whose host element survives React remounts (fullscreen, overlay). */
export interface NeovimTerminal {
  attach: (parent: HTMLElement) => void
  detach: () => void
  fit: () => void
  focus: () => void
  /** Repaint with the colors of the current app theme. */
  restyle: () => void
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

/** Lines picked in visual mode with the ask keymap, 1-based and inclusive, `text` is those full lines of the file at `path`. */
export interface NeovimSelection {
  path: string
  start_line: number
  end_line: number
  text: string
}

/** Sends a control frame to the backend session. `queue` keeps it for the open frame while the socket connects. */
type SendFrame = (frame: Record<string, unknown>, queue: boolean) => boolean

interface NeovimOptions {
  /** Install visual-mode `<leader>a`, which reports the selected lines to `onSelection`. */
  askKeymap?: boolean
  onSelection?: (selection: NeovimSelection) => void
  /** Neovim itself (`:bnext`) put the buffer of another open file on screen. Not called for `openFile`. */
  onSwitched?: (path: string) => void
  /** `openFile(path)` could not load the file, the buffer shown did not change. */
  onOpenFailed?: (path: string, message: string) => void
  /** Takes the PNG of a paste instead of pasting text, resolves to the text to insert below the cursor, null for none. */
  onPasteImage?: (image: Blob) => Promise<string | null>
}

/** The PNG a paste carries, null when the clipboard holds none or also plain text (copied spreadsheet cells come with a picture and paste as text). */
function pastedImage(clipboard: DataTransfer | null): Blob | null {
  if (!clipboard) return null
  const items = Array.from(clipboard.items)
  if (items.some((item) => item.kind === "string" && item.type === "text/plain")) return null
  return items.find((item) => item.kind === "file" && item.type === "image/png")?.getAsFile() ?? null
}

/** Run one backend Neovim session for `document`, which is its first buffer.
 *
 * - `:w` of any buffer calls `save(text, path)`, whose resolved text replaces that buffer if it differs
 * - Yanks (`y`) are copied to the browser clipboard
 * - `:q` with exit code 0 calls `onQuit`
 * - `openFile(path)` shows the buffer of another file, loading it from disk when it is not open yet. The buffer shown before
 *   stays loaded with its unsaved edits, nothing is saved. Called while the session connects it waits for the socket and returns true, false without a session
 * - `setContent(text, path)` hands text that is already on disk to the buffer of `path`, hidden or
 *   shown, false while the session is not open
 *   - An unmodified buffer is replaced (one undo step) and stays unmodified
 *   - A modified buffer gets the text merged into its unsaved edits, the agent's lines win where both changed, and stays modified
 * - `renameFile(from, to)` renames the buffer of `from` (kept while connecting too), keeping its text and unsaved edits. A changed `document.path` under the same `key` renames the buffer of the previous path
 * - `insertLines(text)` inserts lines below the cursor of the shown buffer, false while the session is not open
 * - Unmounting or a new `document.key` discards unsaved edits
 */
export function useNeovimTerminal(document: NeovimDocument | null, save: (text: string, path: string) => Promise<string>, onQuit: () => void, options: NeovimOptions = {}) {
  const [terminal, setTerminal] = useState<NeovimTerminal | null>(null)
  const [status, setStatus] = useState<NeovimStatus>({ state: "running" })
  const [generation, setGeneration] = useState(0)
  const readDocument = useEffectEvent(() => document)
  const readOptions = useEffectEvent(() => options)
  const saveText = useEffectEvent(save)
  const quit = useEffectEvent(onQuit)
  const sendFrameRef = useRef<SendFrame | null>(null)
  const shownRef = useRef<{ key: string; path: string } | null>(null)
  const setContent = useCallback((text: string, path: string) => sendFrameRef.current?.({ type: "set_content", content: text, path }, false) ?? false, [])
  const openFile = useCallback((path: string) => sendFrameRef.current?.({ type: "open_file", path }, true) ?? false, [])
  const renameFile = useCallback((from: string, to: string) => sendFrameRef.current?.({ type: "renamed", from, path: to }, true) ?? false, [])
  const insertLines = useCallback((text: string) => sendFrameRef.current?.({ type: "insert_lines", text }, false) ?? false, [])

  const documentKey = document?.key ?? null
  const documentPath = document?.path ?? null
  useEffect(() => {
    const previous = shownRef.current
    shownRef.current = documentKey === null || documentPath === null ? null : { key: documentKey, path: documentPath }
    if (previous && previous.key === documentKey && documentPath !== null && previous.path !== documentPath) renameFile(previous.path, documentPath)
  }, [documentKey, documentPath, renameFile])

  useEffect(() => {
    const opened = readDocument()
    const withAskKeymap = readOptions().askKeymap ?? false
    if (documentKey === null || !opened) return
    let disposed = false
    let dispose = () => {}

    void (async () => {
      const [{ Terminal }, { FitAddon }, { WebglAddon }] = await Promise.all([import("@xterm/xterm"), import("@xterm/addon-fit"), import("@xterm/addon-webgl")])
      if (disposed) return
      const host = window.document.createElement("div")
      host.className = "h-full w-full font-mono"
      const xterm = new Terminal({ fontSize: FONT_SIZE, cursorBlink: true, theme: terminalTheme(window.document.body).theme })
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

      // Control frames wait for the open frame when the socket is still connecting and `queue` is set, so a file opened or
      // renamed meanwhile is not lost. False when the frame was not taken.
      const queued: string[] = []
      const sendFrame: SendFrame = (frame, queue) => {
        const text = JSON.stringify(frame)
        if (socket.readyState === WebSocket.OPEN) send(text)
        else if (socket.readyState === WebSocket.CONNECTING && queue) queued.push(text)
        else return false
        return true
      }

      // A pasted PNG is uploaded by `onPasteImage` instead of pasted as text, what it returns goes below the cursor.
      // Capturing at the host stops the paste before xterm's own handler on its textarea
      host.addEventListener("paste", (event) => {
        const upload = readOptions().onPasteImage
        const image = upload && pastedImage(event.clipboardData)
        if (!upload || !image) return
        event.preventDefault()
        event.stopPropagation()
        void upload(image).then((text) => { if (text !== null) sendFrame({ type: "insert_lines", text }, false) })
      }, true)

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
        send(JSON.stringify({ type: "open", path: opened.path, filename: opened.filename, content: opened.content, base_content: opened.base_content, rows: xterm.rows, cols: xterm.cols, ask_keymap: withAskKeymap }))
        queued.splice(0).forEach(send)
        if (backgroundIsLight !== null) sendBackground(backgroundIsLight)
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
        if (message.type === "selection") { readOptions().onSelection?.({ path: message.path, start_line: message.start_line, end_line: message.end_line, text: message.text }); return }
        if (message.type === "switched") { readOptions().onSwitched?.(message.path); return }
        if (message.type === "open_failed") { readOptions().onOpenFailed?.(message.path, message.message); return }
        if (message.type !== "save") return
        try {
          send(JSON.stringify({ type: "saved", id: message.id, content: await saveText(message.content, message.path) }))
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

      // Neovim's 'background' follows the theme behind the terminal, set by the backend over RPC. Its own startup
      // query (OSC 11) races the slot's styling and misses a theme picked before reconnecting
      let backgroundIsLight: boolean | null = null
      const sendBackground = (light: boolean) => send(JSON.stringify({ type: "background", light }))
      const applyTheme = (parent: HTMLElement) => {
        const { theme, isLight } = terminalTheme(parent)
        xterm.options.theme = theme
        if (isLight !== backgroundIsLight) sendBackground(isLight)
        backgroundIsLight = isLight
      }

      setStatus({ state: "running" })
      setTerminal({
        attach: (parent) => { parent.appendChild(host); applyTheme(parent); fit() },
        detach: () => host.remove(),
        fit,
        restyle: () => { if (host.parentElement) applyTheme(host.parentElement) },
        focus: () => xterm.focus(),
      })
      sendFrameRef.current = sendFrame
      dispose = () => { sendFrameRef.current = null; if (pendingResize) clearTimeout(pendingResize.timer); socket.close(); xterm.dispose() }
    })()

    return () => {
      disposed = true
      dispose()
      setTerminal(null)
    }
  }, [documentKey, generation])

  return { terminal, status, restart: () => setGeneration((value) => value + 1), setContent, openFile, renameFile, insertLines }
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
  const theme = useThemeName()

  useEventListener(slotRef, "keydown", (event) => event.stopPropagation())

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

  useEffect(() => { terminal?.restyle() }, [terminal, theme])

  return (
    <div ref={slotRef} data-neovim-terminal className={cn("relative min-h-0 flex-1 overflow-hidden bg-page px-2 pt-1", className)}>
      {status.state === "stopped" && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-page text-xs text-ink-faint">
          <span>{status.message}</span>
          <button onClick={onRestart} className="rounded px-2 py-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
            Restart Neovim
          </button>
        </div>
      )}
    </div>
  )
}
