"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react"
import "@xterm/xterm/css/xterm.css"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useEventListener } from "@/hooks/useEventListener"
import { useThemeName } from "@/hooks/useThemeName"
import { terminalSessionUrl } from "@/lib/backend/urls"
import { isEditableTarget } from "@/lib/utils"
import { NERD_FONTS, TERMINAL_FONT_SIZE, terminalTheme } from "./terminalTheme"

const POLICY_VIOLATION = 1008
/** A second Esc this soon hides the window even while a program (vim, less) took the first one. */
const DOUBLE_ESCAPE_MS = 400

/** A running shell whose xterm host survives the window closing. */
interface ShellTerminal {
  attach: (parent: HTMLElement) => void
  detach: () => void
  fit: () => void
  focus: () => void
  restyle: () => void
}

/** A bare `t` typed outside text fields (composers, inputs, terminals) that nothing else took. */
function isOpenTerminalKey(event: KeyboardEvent): boolean {
  const hasModifier = event.ctrlKey || event.metaKey || event.altKey
  return event.key === "t" && !hasModifier && !event.repeat && !event.isComposing && !event.defaultPrevented && !isEditableTarget(event.target, { includeSelect: true })
}

/** A bare Esc, the key that leaves the terminal. */
function isLeaveKey(event: KeyboardEvent): boolean {
  return event.key === "Escape" && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey && !event.isComposing
}

/** Run one backend login shell, a new `sessionKey` starts a new one.
 *
 * - `onExit` follows the shell's `exit`
 * - `onLeave` follows an Esc at the shell's prompt, or a double Esc while a program runs (the program gets the first one)
 */
function useShellSession(sessionKey: number, onExit: () => void, onLeave: () => void) {
  const [terminal, setTerminal] = useState<ShellTerminal | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const exit = useEffectEvent(onExit)
  const leave = useEffectEvent(onLeave)

  useEffect(() => {
    let disposed = false
    let dispose = () => {}

    void (async () => {
      const [{ Terminal }, { FitAddon }, { WebglAddon }] = await Promise.all([import("@xterm/xterm"), import("@xterm/addon-fit"), import("@xterm/addon-webgl")])
      if (disposed) return
      const host = window.document.createElement("div")
      host.className = "h-full w-full font-mono"
      const xterm = new Terminal({ fontSize: TERMINAL_FONT_SIZE, cursorBlink: true, theme: terminalTheme(window.document.body).theme })
      const fitAddon = new FitAddon()
      xterm.loadAddon(fitAddon)
      let isOpen = false
      let exited = false

      const socket = new WebSocket(terminalSessionUrl())
      socket.binaryType = "arraybuffer"
      const send = (data: string | Uint8Array<ArrayBuffer>) => { if (socket.readyState === WebSocket.OPEN) socket.send(data) }
      const encoder = new TextEncoder()
      socket.onopen = () => send(JSON.stringify({ type: "open", rows: xterm.rows, cols: xterm.cols }))
      socket.onmessage = (event: MessageEvent<ArrayBuffer | string>) => {
        if (typeof event.data !== "string") { xterm.write(new Uint8Array(event.data)); return }
        const message = JSON.parse(event.data)
        if (message.type === "exit") exited = true
        else if (message.type === "leave") leave()
      }
      socket.onclose = (event) => {
        if (disposed) return
        if (exited) exit()
        else setFailure(event.code === POLICY_VIOLATION ? "The terminal is not enabled for this account." : "Lost the connection to the terminal.")
      }
      xterm.onData((data) => send(encoder.encode(data)))
      xterm.onBinary((data) => send(Uint8Array.from(data, (char) => char.charCodeAt(0))))
      xterm.onResize(({ rows, cols }) => send(JSON.stringify({ type: "resize", rows, cols })))
      // Esc is the backend's call: at the prompt it leaves, while a program runs the program gets it
      let lastEscape = -Infinity
      xterm.attachCustomKeyEventHandler((event) => {
        if (!isLeaveKey(event)) return true
        event.preventDefault()
        if (event.type !== "keydown" || event.repeat) return false
        const isDouble = event.timeStamp - lastEscape < DOUBLE_ESCAPE_MS
        if (isDouble) leave()
        else send(JSON.stringify({ type: "escape" }))
        lastEscape = isDouble ? -Infinity : event.timeStamp
        return false
      })

      // xterm measures cells on open, so open only once the host is visible. Canvas cannot resolve CSS variables, so the app font is the computed one.
      const fit = () => {
        if (host.clientWidth === 0 || host.clientHeight === 0) return
        if (!isOpen) {
          xterm.options.fontFamily = `${NERD_FONTS}, ${getComputedStyle(host).fontFamily}`
          xterm.open(host)
          isOpen = true
          try {
            const webgl = new WebglAddon()
            webgl.onContextLoss(() => webgl.dispose())
            xterm.loadAddon(webgl)
          } catch { /* no WebGL2, the DOM renderer stays */ }
        }
        fitAddon.fit()
      }
      const restyle = () => { if (host.parentElement) xterm.options.theme = terminalTheme(host.parentElement).theme }

      setFailure(null)
      setTerminal({
        attach: (parent) => { parent.appendChild(host); restyle(); fit() },
        detach: () => host.remove(),
        fit,
        focus: () => xterm.focus(),
        restyle,
      })
      dispose = () => { socket.close(); xterm.dispose() }
    })()

    return () => {
      disposed = true
      dispose()
      setTerminal(null)
    }
  }, [sessionKey])

  return { terminal, failure }
}

/** Show `terminal` here. Keys stop at this element, so app shortcuts never steal them. */
function ShellTerminalSlot({ terminal, failure, onRestart }: { terminal: ShellTerminal | null; failure: string | null; onRestart: () => void }) {
  const slotRef = useRef<HTMLDivElement>(null)
  const theme = useThemeName()

  useEventListener(slotRef, "keydown", (event) => event.stopPropagation())

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

  useEffect(() => { terminal?.restyle() }, [terminal, theme])

  return (
    <div ref={slotRef} data-terminal className="relative min-h-0 flex-1 overflow-hidden">
      {failure && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-paper text-xs text-ink-faint">
          <span>{failure}</span>
          <button onClick={onRestart} className="rounded px-2 py-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
            Restart terminal
          </button>
        </div>
      )}
    </div>
  )
}

/** One tab's shell: `sessionKey` names its run, the window shows while `shown`. Unmounting hangs the shell up. */
function TabShell({ sessionKey, shown, onExit, onHide, onRestart }: {
  sessionKey: number
  shown: boolean
  onExit: () => void
  onHide: () => void
  onRestart: () => void
}) {
  const { terminal, failure } = useShellSession(sessionKey, onExit, onHide)
  if (!shown) return null
  return (
    <FloatingWindow onClose={onHide} label="Terminal" panelClassName="h-[66.667vh] max-h-full w-[66.667vw] max-w-full p-3">
      <ShellTerminalSlot terminal={terminal} failure={failure} onRestart={onRestart} />
    </FloatingWindow>
  )
}

/** A tab's shell: `key` names its run (a restart counts up), `visible` whether its window is open while the tab is frontmost. */
interface TabShellState {
  key: number
  visible: boolean
}

/** Floating login shells on the backend host, one per tab, for logins with host access.
 *
 * - `t` outside a text field opens the active tab's shell, 2/3 of the screen wide and high. Another tab shows its own shell, or none
 * - Esc at the prompt or a click beside it hides it, the shell keeps running and `t` shows it again. While a program runs (vim,
 *   less) Esc is the program's, a double Esc hides the window
 * - `exit` ends the shell and closes the window, closing its tab hangs it up
 */
export function ShellTerminals({ activeTabId, tabIds }: { activeTabId: string | null; tabIds: readonly string[] }) {
  const hostAccess = useHostAccess()
  const [shells, setShells] = useState<ReadonlyMap<string, TabShellState>>(new Map())
  // Shells of closed tabs leave the state here, their windows already unmounted with the tab
  const update = useCallback((tabId: string, change: (shell: TabShellState | undefined) => TabShellState | undefined) => {
    setShells((current) => {
      const next = new Map([...current].filter(([id]) => tabIds.includes(id)))
      const changed = change(current.get(tabId))
      if (changed) next.set(tabId, changed)
      else next.delete(tabId)
      return next
    })
  }, [tabIds])

  useEventListener("document", "keydown", (event) => {
    if (!hostAccess || activeTabId === null || !isOpenTerminalKey(event)) return
    event.preventDefault()
    update(activeTabId, (shell) => ({ key: shell?.key ?? 0, visible: true }))
  })

  return [...shells].filter(([tabId]) => tabIds.includes(tabId)).map(([tabId, shell]) => (
    <TabShell
      key={tabId}
      sessionKey={shell.key}
      shown={shell.visible && tabId === activeTabId}
      onExit={() => update(tabId, () => undefined)}
      onHide={() => update(tabId, (current) => current && { ...current, visible: false })}
      onRestart={() => update(tabId, (current) => current && { ...current, key: current.key + 1 })}
    />
  ))
}
