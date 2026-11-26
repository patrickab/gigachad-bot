"use client"

import { createContext, useCallback, useEffect, useEffectEvent, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { X } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { PanelToggleButton } from "@/components/ui/PanelToggleButton"
import { SidePanel } from "@/components/ui/SidePanel"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import { writeRootFile } from "@/lib/backend/api"
import { baseName, extOf, kindOf } from "@/lib/paths"
import { fenceCode } from "@gigachad/chat-ui"
import { NeovimEditor, useNeovimTerminal } from "./NeovimEditor"
import { RawFileView } from "./RawFileView"
import { TEXT_CODEC, useRootFileDocument, type FileLoadFailure } from "@/hooks/useRootFileDocument"

/** Panel share of the width it splits with the main surface, on every open. */
const OPEN_SHARE = 0.5
/** Widest drag, as a share of that width, so the main surface stays usable. */
const MAX_SHARE = 0.8

interface FilePanelContextValue {
  slot: HTMLElement | null
  setSlot: (slot: HTMLElement | null) => void
  ownerId: string | null
  /** Open width the owner asked for, null for the half share. */
  ownerWidth: number | null
  claim: (id: string, close: () => void, width: number | null) => void
  release: (id: string) => void
  openFile: (path: string, line?: number) => void
  collapsed: boolean
  toggleCollapsed: () => void
}

const FilePanelContext = createContext<FilePanelContextValue | null>(null)

function useFilePanelContext(): FilePanelContextValue {
  return useRequiredContext(FilePanelContext, "File panel used outside FilePanelProvider")
}

/** Open an absolute file path in this tab's file panel, at 1-based `line` when given (in Neovim). */
export function useOpenFile(): (path: string, line?: number) => void {
  return useFilePanelContext().openFile
}

/**
 * Own one tab's right file panel, which shares the width with the chat.
 *
 * - One `FilePanel` shows at a time, opening another closes the shown one
 * - `openFile` shows a file root path, e.g. a file link in a Pi answer
 * - Collapsing hides the panel and keeps its content (Neovim keeps running), opening something expands it
 */
export function FilePanelProvider({ children }: { children: ReactNode }) {
  const [slot, setSlot] = useState<HTMLElement | null>(null)
  const [ownerId, setOwnerId] = useState<string | null>(null)
  const [ownerWidth, setOwnerWidth] = useState<number | null>(null)
  const [filePath, setFilePath] = useState<string | null>(null)
  // A new object per open: opening the shown file at a line moves its cursor again
  const [fileLine, setFileLine] = useState<{ line: number } | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const ownerRef = useRef<{ id: string; close: () => void } | null>(null)

  const claim = useCallback((id: string, close: () => void, width: number | null) => {
    const previous = ownerRef.current
    ownerRef.current = { id, close }
    setOwnerId(id)
    setOwnerWidth(width)
    setCollapsed(false)
    if (previous && previous.id !== id) previous.close()
  }, [])
  const release = useCallback((id: string) => {
    if (ownerRef.current?.id !== id) return
    ownerRef.current = null
    setOwnerId(null)
  }, [])
  const closeFile = useCallback(() => setFilePath(null), [])
  const toggleCollapsed = useCallback(() => setCollapsed((value) => !value), [])
  // Reopening the shown file claims nothing, so expand here too
  const openFile = useCallback((path: string, line?: number) => {
    setFilePath(path)
    setFileLine(line !== undefined && Number.isInteger(line) && line > 0 ? { line } : null)
    setCollapsed(false)
  }, [])

  const value = useMemo(
    () => ({ slot, setSlot, ownerId, ownerWidth, claim, release, openFile, collapsed, toggleCollapsed }),
    [slot, ownerId, ownerWidth, claim, release, openFile, collapsed, toggleCollapsed],
  )
  return (
    <FilePanelContext.Provider value={value}>
      {children}
      {filePath && <RootFileView key={filePath} path={filePath} at={fileLine} onClose={closeFile} />}
    </FilePanelContext.Provider>
  )
}

/**
 * The panel itself, placed right after the tab's main surface: the right `SidePanel`.
 *
 * - Opens at half the width the two share, keeps a dragged width until closed, never wider than the drag limit
 * - Closed or collapsed it eases shut and stays mounted, so the slot keeps its portal content
 */
export function FilePanelSlot() {
  const { slot, setSlot, ownerId, ownerWidth, collapsed } = useFilePanelContext()
  const [sharedWidth, setSharedWidth] = useState(0)
  const [openWidth, setOpenWidth] = useState(0)
  const shown = ownerId !== null
  const expanded = shown && !collapsed

  useLayoutEffect(() => {
    const panel = slot?.closest("aside")
    const main = panel?.previousElementSibling
    if (!shown || !panel || !main) return
    const measureShared = () => Math.round(main.getBoundingClientRect().width + panel.getBoundingClientRect().width)
    const opened = measureShared()
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the open width is measured from the laid-out surfaces
    setSharedWidth(opened)
    setOpenWidth(ownerWidth ?? Math.round(opened * OPEN_SHARE))
    // A window or left sidebar resize changes the shared width, the panel shrinks into the new limit. A hidden tab measures 0, its width stays
    const observer = new ResizeObserver(() => {
      const shared = measureShared()
      if (shared === 0) return
      setSharedWidth(shared)
      setOpenWidth((width) => Math.min(width, Math.round(shared * MAX_SHARE)))
    })
    observer.observe(main)
    return () => observer.disconnect()
  }, [shown, slot, ownerWidth])

  return (
    <SidePanel side="right" open={expanded} width={openWidth} onWidthChange={setOpenWidth} maxWidth={Math.round(sharedWidth * MAX_SHARE)} label="File panel">
      <div ref={setSlot} className="flex min-h-0 flex-1 flex-col" />
    </SidePanel>
  )
}

/** Top bar toggle reopening the collapsed file panel. Expanded, the panel's own corner holds the toggle. */
export function FilePanelToggle() {
  const { ownerId, collapsed, toggleCollapsed } = useFilePanelContext()
  if (ownerId === null || !collapsed) return null
  return <PanelToggleButton open={false} onClick={toggleCollapsed} openLabel="Open file panel" closeLabel="Close file panel" />
}

/**
 * Show `children` in the tab's file panel. The header holds the collapse toggle in the panel's top-left corner, then `label`, `actions` and a close button.
 *
 * - `width` px replaces the half-share open width
 * - `bare` keeps only the toggle and `actions`, the toggle then closes through `onClose`, for owners with their own reopen control (PDF chat, workbench chat)
 */
export function FilePanel({ label, onClose, actions, width, bare = false, children }: {
  label: string
  onClose: () => void
  actions?: ReactNode
  width?: number
  bare?: boolean
  children: ReactNode
}) {
  const { slot, ownerId, claim, release, toggleCollapsed } = useFilePanelContext()
  const id = useId()
  const closeOwned = useEffectEvent(onClose)
  useEffect(() => {
    claim(id, () => closeOwned(), width ?? null)
    return () => release(id)
  }, [id, claim, release, width])

  if (ownerId !== id || !slot) return null
  return createPortal(
    <section aria-label={label} className="flex min-h-0 flex-1 flex-col">
      {/* Insets mirror the left sidebar's header */}
      <div className="flex h-10 shrink-0 items-center gap-2 pl-4 pr-2 text-[11px] text-ink-subtle">
        <PanelToggleButton open onClick={bare ? onClose : toggleCollapsed} openLabel="Open file panel" closeLabel={bare ? `Close ${label.toLowerCase()}` : "Close file panel"} />
        {bare ? actions : <>
          <span className="min-w-0 flex-1 truncate">{label}</span>
          {actions}
          <button type="button" aria-label="Close panel" onClick={onClose} className="rounded-md p-1 text-ink-subtle transition-colors hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
            <X className="h-3.5 w-3.5" />
          </button>
        </>}
      </div>
      {children}
    </section>,
    slot,
  )
}

export const LOAD_MESSAGES: Record<FileLoadFailure, string> = {
  forbidden: "This file is outside the folders gigachad-bot can read.",
  missing: "This file does not exist.",
  binary: "This is not a text file.",
  failed: "Could not load this file.",
}

/** Markdown renders as itself, any other text as a code block highlighted by its extension. */
function asMarkdown(path: string, content: string): string {
  return kindOf(path) === "markdown" ? content : fenceCode(extOf(path), content)
}

/** Show `path` (absolute, inside a file root): image and PDF as such, text in Neovim (setting on) or read-only. */
function RootFileView({ path, at, onClose }: { path: string; at: { line: number } | null; onClose: () => void }) {
  const kind = kindOf(path)
  if (kind === "image" || kind === "pdf") return <FilePanel label={path} onClose={onClose}><RawFileView path={path} kind={kind} /></FilePanel>
  return <TextFileView path={path} at={at} onClose={onClose} />
}

/** Text file: in Neovim `:w` saves it in place and `:q` closes the panel, `at` puts the cursor on its line. */
function TextFileView({ path, at, onClose }: { path: string; at: { line: number } | null; onClose: () => void }) {
  const neovim = useNeovimSetting()
  const write = useCallback((text: string) => writeRootFile(path, text), [path])
  // While Neovim runs it holds the text, so a change on disk is merged into its buffer
  const { value: content, loadError, reload, saveFromNeovim } = useRootFileDocument({
    path, codec: TEXT_CODEC, write, blankWhenMissing: false, onFollowed: (text, changed) => { neovimSession.send({ type: "set_content", content: text, path: changed }) },
  })
  useEffect(() => { reload() }, [reload])
  const neovimSession = useNeovimTerminal(neovim.enabled && content !== null ? { key: path, path, filename: baseName(path), content } : null, saveFromNeovim, onClose)
  // Once the session takes commands (its terminal is set), queued until its socket opens
  const { terminal, send } = neovimSession
  useEffect(() => {
    if (terminal && at) send({ type: "open_file", path, line: at.line })
  }, [terminal, at, path, send])

  return (
    <FilePanel label={path} onClose={onClose}>
      {loadError ? (
        <div className="flex flex-1 items-center justify-center text-xs text-ink-faint">{LOAD_MESSAGES[loadError]}</div>
      ) : content === null ? (
        <div className="flex flex-1 items-center justify-center text-xs text-ink-faint">Loading…</div>
      ) : neovim.enabled ? (
        <NeovimEditor terminal={neovimSession.terminal} status={neovimSession.status} onRestart={neovimSession.restart} />
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
          <LaTeXMarkdown content={asMarkdown(path, content)} />
        </div>
      )}
    </FilePanel>
  )
}
