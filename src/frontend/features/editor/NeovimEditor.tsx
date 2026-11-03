"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react"
import "@xterm/xterm/css/xterm.css"
import { cn, errorMessage } from "@/lib/utils"
import { neovimSessionUrl } from "@/lib/backend/urls"
import { readClipboardForPaste, storeClipboardImage } from "./clipboardPaste"
import { useEventListener } from "@/hooks/useEventListener"
import { useThemeName } from "@/hooks/useThemeName"
import type { CellCursor, CellExitDirection, CellText, CellTextMetrics } from "@/features/notebook/cellTerminal"
import { NERD_FONTS, TERMINAL_FONT_SIZE, terminalTheme } from "./terminalTheme"

const POLICY_VIOLATION = 1008
/** Quiet time after the last size change before the terminal refits. */
const FIT_SETTLE_MS = 150
/** Longest wait for Neovim's redraw after a resize before xterm resizes anyway. */
const REDRAW_TIMEOUT_MS = 500
// Synchronized output (DEC mode 2026): xterm holds rendering in between, Neovim wraps each redraw in it.
const SYNC_START = "\x1b[?2026h"
const SYNC_END = "\x1b[?2026l"
// Shift+Enter and Ctrl+Enter as the kitty keyboard protocol writes them: key 13 with the shift (2) or ctrl (5) modifier
const SHIFT_ENTER = "\x1b[13;2u"
const CTRL_ENTER = "\x1b[13;5u"
// One char per byte, so the ASCII marker can be found in raw PTY output
const BYTE_DECODER = new TextDecoder("latin1")

/** A running terminal whose host element survives React remounts (fullscreen, overlay). */
export interface NeovimTerminal {
  attach: (parent: HTMLElement) => void
  detach: () => void
  fit: () => void
  focus: () => void
  /** Repaint with the colors of the current app theme. */
  restyle: () => void
  /** How the terminal draws text (font, size of one cell in px), null before it was on screen. Kept while the terminal is off screen. */
  metrics: () => CellTextMetrics | null
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
  /** The path of the buffer on screen, what a pasted image is stored next to. The document's own path without it, right while the session shows one buffer. */
  shownPath?: () => string
  /** Shift+Enter and Ctrl+Enter reach Neovim as the CSI-u sequences `<S-CR>` and `<C-CR>` (xterm sends a plain Enter for both), what the notebook buffers map to "run cell". */
  cellMode?: boolean
  /** The cell `id` holds `text` after a change in Neovim. Notifications arrive in the order Neovim made them. */
  onCellEdit?: (id: string, text: string) => void
  /** Neovim moved its cursor into the cell `id` (`j`/`k`, `G`, a search, an LSP jump). */
  onCellFocus?: (id: string) => void
  /** Shift+Enter (`advance`) or Ctrl+Enter in the cell `id`, with the text it had. */
  onCellRun?: (id: string, text: string, advance: boolean) => void
  /** Esc in normal mode in the markdown cell `id`. */
  onCellLeave?: (id: string) => void
  /** Neovim's cursor left the cell `id` past its edge toward a cell of the other kind. */
  onCellExit?: (id: string, direction: CellExitDirection) => void
  /** `:w`: nothing is answered, the notebook is saved. */
  onCellSave?: () => void
  /** Neovim drew the cell `id` (`cell_render`, unchecked). */
  onCellRender?: (id: string, rendering: unknown) => void
  /** The text of the cell `id` Neovim's window is pinned to takes `rows` screen rows there. */
  onCellHeight?: (id: string, rows: number) => void
}

/** Move focus from the terminal in `host` to the closest focusable element around it (a notebook cell takes its arrow keys again), else to the page. */
function giveFocusBack(host: HTMLElement) {
  const around = host.parentElement?.closest<HTMLElement>("[tabindex]")
  if (around) around.focus({ preventScroll: true })
  else if (window.document.activeElement instanceof HTMLElement) window.document.activeElement.blur()
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
 * - Esc in normal mode gives focus back to the app (`leave`): the closest focusable element around the terminal, else the page
 * - `:q` with exit code 0 calls `onQuit`
 * - `openFile(path)` shows the buffer of another file, loading it from disk when it is not open yet. The buffer shown before
 *   stays loaded with its unsaved edits, nothing is saved. Called while the session connects it waits for the socket and returns true, false without a session
 * - `setContent(text, path)` hands text that is already on disk to the buffer of `path`, hidden or
 *   shown. While the session is opening the latest text per path waits for the open frame and the call returns true, false without a session
 *   - An unmodified buffer is replaced (one undo step) and stays unmodified
 *   - A modified buffer gets the text merged into its unsaved edits, the agent's lines win where both changed, and stays modified
 * - `renameFile(from, to)` renames the buffer of `from` (kept while connecting too), keeping its text and unsaved edits. A changed `document.path` under the same `key` renames the buffer of the previous path
 * - `insertLines(text)` inserts lines below the cursor of the shown buffer, false while the session is not open
 * - Neovim's `p` and `"+p` ask the browser clipboard (`clipboard_read`). A screenshot is stored in `imgs/` next to the shown LaTeX or Markdown file and pasted as its reference, text as text, a failure is shown in Neovim. Ctrl+Shift+V of a screenshot does the same through `insert_lines`
 * - Unmounting or a new `document.key` discards unsaved edits
 * - `setCells(filename, cells)` hands the notebook's code and markdown cells in order (a buffer per kind, no echo back, ids unique, `filename` gives the code buffer's extension when Neovim creates it), `focusCell(id, cursor)` pins the window to a cell. Edits, Shift+Enter and Ctrl+Enter, Esc in a markdown cell, `:w`, the cursor moving into another cell or out of its buffer's cells come back as `onCellEdit`, `onCellRun`, `onCellLeave`, `onCellSave`, `onCellFocus` and `onCellExit` in order, how Neovim draws each code cell as `onCellRender`, the screen rows of the pinned cell as `onCellHeight`
 * - `cellMode` makes the session a notebook's: it opens without the document's filename and content (only its `path`, which names the buffers beside the notebook when absolute and in a file root, and picks the working directory) and sends Shift+Enter as `<S-CR>`, Ctrl+Enter as `<C-CR>`
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
  // The text `setContent` took while the session was opening, by path. Null when no session is opening
  const pendingContentRef = useRef<Map<string, string> | null>(null)
  const setContent = useCallback((text: string, path: string) => {
    if (sendFrameRef.current?.({ type: "set_content", content: text, path }, false)) return true
    if (!pendingContentRef.current) return false
    pendingContentRef.current.set(path, text)
    return true
  }, [])
  const openFile = useCallback((path: string) => sendFrameRef.current?.({ type: "open_file", path }, true) ?? false, [])
  const renameFile = useCallback((from: string, to: string) => sendFrameRef.current?.({ type: "renamed", from, path: to }, true) ?? false, [])
  const insertLines = useCallback((text: string) => sendFrameRef.current?.({ type: "insert_lines", text }, false) ?? false, [])
  const setCells = useCallback((filename: string, cells: readonly CellText[]) => sendFrameRef.current?.({ type: "set_cells", filename, cells }, true) ?? false, [])
  const focusCell = useCallback((id: string, cursor: CellCursor) => sendFrameRef.current?.({ type: "focus_cell", id, cursor }, true) ?? false, [])

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
    const inCellMode = readOptions().cellMode ?? false
    if (documentKey === null || !opened) return
    let disposed = false
    let dispose = () => {}
    const pendingContent = new Map<string, string>()
    pendingContentRef.current = pendingContent

    void (async () => {
      const [{ Terminal }, { FitAddon }, { WebglAddon }] = await Promise.all([import("@xterm/xterm"), import("@xterm/addon-fit"), import("@xterm/addon-webgl")])
      if (disposed) return
      const host = window.document.createElement("div")
      host.className = "h-full w-full font-mono"
      const xterm = new Terminal({ fontSize: TERMINAL_FONT_SIZE, cursorBlink: true, theme: terminalTheme(window.document.body).theme })
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

      // A pasted PNG (Ctrl+Shift+V) is stored next to the shown file and its reference inserted below the cursor instead of pasted as text.
      // Capturing at the host stops the paste before xterm's own handler on its textarea
      host.addEventListener("paste", (event) => {
        const image = pastedImage(event.clipboardData)
        if (!image) return
        event.preventDefault()
        event.stopPropagation()
        storeClipboardImage(readOptions().shownPath?.() ?? readDocument()?.path ?? "", image).then(
          (reference) => sendFrame({ type: "insert_lines", text: reference }, false),
          (error) => sendFrame({ type: "message", text: errorMessage(error, "Could not store the image") }, false),
        )
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
        const rowsAndCols = { rows: xterm.rows, cols: xterm.cols, ask_keymap: withAskKeymap }
        send(JSON.stringify(inCellMode ? { type: "open", cell_mode: true, path: opened.path, ...rowsAndCols } : { type: "open", path: opened.path, filename: opened.filename, content: opened.content, base_content: opened.base_content, ...rowsAndCols }))
        queued.splice(0).forEach(send)
        for (const [path, content] of pendingContent) send(JSON.stringify({ type: "set_content", content, path }))
        pendingContent.clear()
        pendingContentRef.current = null
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
        if (message.type === "leave") { giveFocusBack(host); return }
        if (message.type === "open_failed") { readOptions().onOpenFailed?.(message.path, message.message); return }
        if (message.type === "cell_edit") { readOptions().onCellEdit?.(message.id, message.text); return }
        if (message.type === "cell_focus") { readOptions().onCellFocus?.(message.id); return }
        if (message.type === "cell_run") { readOptions().onCellRun?.(message.id, message.text, message.advance !== false); return }
        if (message.type === "cell_leave") { readOptions().onCellLeave?.(message.id); return }
        if (message.type === "cell_exit") { readOptions().onCellExit?.(message.id, message.direction === "up" ? "up" : "down"); return }
        if (message.type === "save_notebook") { readOptions().onCellSave?.(); return }
        if (message.type === "cell_render") { readOptions().onCellRender?.(message.id, message); return }
        if (message.type === "cell_height") { readOptions().onCellHeight?.(message.id, message.rows); return }
        if (message.type === "clipboard_read") {
          try {
            send(JSON.stringify({ type: "clipboard", id: message.id, text: await readClipboardForPaste(message.path) }))
          } catch (error) {
            send(JSON.stringify({ type: "clipboard", id: message.id, error: errorMessage(error, "Could not read the clipboard") }))
          }
          return
        }
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
      // xterm sends a plain Enter for Shift+Enter and Ctrl+Enter, which Neovim cannot tell from Enter. The CSI-u form is what a terminal with the kitty keyboard protocol sends
      xterm.attachCustomKeyEventHandler((event) => {
        if (!readOptions().cellMode || event.key !== "Enter" || event.shiftKey === event.ctrlKey || event.altKey || event.metaKey) return true
        event.preventDefault()
        if (event.type === "keydown") send(encoder.encode(event.shiftKey ? SHIFT_ENTER : CTRL_ENTER))
        return false
      })
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

      // The last size measured on screen: a terminal that is off screen measures nothing. The same object while nothing changed
      let measured: CellTextMetrics | null = null
      const metrics = () => {
        const screen = isOpen ? xterm.element?.querySelector<HTMLElement>(".xterm-screen") : null
        const box = screen?.getBoundingClientRect()
        if (!box || box.height === 0 || xterm.rows === 0 || xterm.cols === 0) return measured
        const next = { fontFamily: xterm.options.fontFamily ?? "", fontSize: TERMINAL_FONT_SIZE, cellWidth: box.width / xterm.cols, cellHeight: box.height / xterm.rows, columns: xterm.cols }
        if (!measured || Object.entries(next).some(([key, value]) => measured?.[key as keyof CellTextMetrics] !== value)) measured = next
        return measured
      }

      setStatus({ state: "running" })
      setTerminal({
        attach: (parent) => { parent.appendChild(host); applyTheme(parent); fit() },
        detach: () => host.remove(),
        fit,
        restyle: () => { if (host.parentElement) applyTheme(host.parentElement) },
        focus: () => xterm.focus(),
        metrics,
      })
      sendFrameRef.current = sendFrame
      dispose = () => { sendFrameRef.current = null; if (pendingResize) clearTimeout(pendingResize.timer); socket.close(); xterm.dispose() }
    })()

    return () => {
      disposed = true
      pendingContentRef.current = null
      dispose()
      setTerminal(null)
    }
  }, [documentKey, generation])

  const restart = useCallback(() => setGeneration((value) => value + 1), [])
  return { terminal, status, restart, setContent, openFile, renameFile, insertLines, setCells, focusCell }
}

function stoppedMessage(exitCode: number | null, closeCode: number): string {
  if (exitCode !== null) return `Neovim exited with code ${exitCode}.`
  if (closeCode === POLICY_VIOLATION) return "Neovim is not enabled for this account."
  return "Lost the connection to Neovim. Unsaved changes are gone."
}

/** Show `terminal` here. Keys stop at this element, so app shortcuts never steal them. */
export function NeovimEditor({ terminal, status, onRestart, className, fitHeightAtOnce = false }: {
  terminal: NeovimTerminal | null
  status: NeovimStatus
  onRestart: () => void
  className?: string
  /** Refit the moment only the height changed, for a slot whose height follows its text: waiting for the size to settle leaves Neovim a row short while the text grows. */
  fitHeightAtOnce?: boolean
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
    let width = slot.clientWidth
    const observer = new ResizeObserver(() => {
      clearTimeout(fitTimer)
      const widthKept = slot.clientWidth === width
      width = slot.clientWidth
      if (fitHeightAtOnce && widthKept) terminal.fit()
      else fitTimer = setTimeout(() => terminal.fit(), FIT_SETTLE_MS)
    })
    observer.observe(slot)
    terminal.focus()
    return () => {
      clearTimeout(fitTimer)
      observer.disconnect()
      terminal.detach()
    }
  }, [terminal, fitHeightAtOnce])

  useEffect(() => { terminal?.restyle() }, [terminal, theme])

  return (
    <div ref={slotRef} data-terminal className={cn("relative min-h-0 flex-1 overflow-hidden bg-page px-2 pt-1", className)}>
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
