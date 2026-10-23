"use client"

import { createContext, useCallback, useContext, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { PanelRight, PanelRightClose, X } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"
import { SidePanelFrame } from "@/components/ui/SidePanel"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { writeRootFile } from "@/lib/backend/api"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { baseName, extOf, kindOf } from "@/lib/paths"
import { cn } from "@/lib/utils"
import { fenceCode } from "@gigachad/chat-ui"
import { NeovimEditor, useNeovimTerminal } from "./NeovimEditor"
import { useRootFileDocument, type FileLoadFailure } from "./useRootFileDocument"

/** Panel share of the width it splits with the main surface, on every open. */
const OPEN_SHARE = 0.5
/** Widest drag, as a share of that width, so the main surface stays usable. */
const MAX_SHARE = 0.8

interface FilePanelContextValue {
  slot: HTMLElement | null
  setSlot: (slot: HTMLElement | null) => void
  ownerId: string | null
  claim: (id: string, close: () => void) => void
  release: (id: string) => void
  openFile: (path: string) => void
  collapsed: boolean
  toggleCollapsed: () => void
}

const FilePanelContext = createContext<FilePanelContextValue | null>(null)

function useFilePanelContext(): FilePanelContextValue {
  const context = useContext(FilePanelContext)
  if (!context) throw new Error("File panel used outside FilePanelProvider")
  return context
}

/** Open an absolute file path in this tab's file panel. */
export function useOpenFile(): (path: string) => void {
  return useFilePanelContext().openFile
}

/**
 * Own one tab's right file panel, which shares the width with the chat (ompweb's right panel).
 *
 * - One `FilePanel` shows at a time, opening another closes the shown one
 * - `openFile` shows a file root path, e.g. a file link from the omp frame
 * - Collapsing hides the panel and keeps its content (Neovim keeps running), opening something expands it
 */
export function FilePanelProvider({ children }: { children: ReactNode }) {
  const [slot, setSlot] = useState<HTMLElement | null>(null)
  const [ownerId, setOwnerId] = useState<string | null>(null)
  const [filePath, setFilePath] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const ownerRef = useRef<{ id: string; close: () => void } | null>(null)

  const claim = useCallback((id: string, close: () => void) => {
    const previous = ownerRef.current
    ownerRef.current = { id, close }
    setOwnerId(id)
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
  const openFile = useCallback((path: string) => {
    setFilePath(path)
    setCollapsed(false)
  }, [])

  const value = useMemo(
    () => ({ slot, setSlot, ownerId, claim, release, openFile, collapsed, toggleCollapsed }),
    [slot, ownerId, claim, release, openFile, collapsed, toggleCollapsed],
  )
  return (
    <FilePanelContext.Provider value={value}>
      {children}
      {filePath && <RootFileView key={filePath} path={filePath} onClose={closeFile} />}
    </FilePanelContext.Provider>
  )
}

/**
 * The panel itself, placed right after the tab's main surface, styled like the left sidebar.
 *
 * - Opens at half the width the two share, keeps a dragged width until closed, never wider than the drag limit
 * - Closed or collapsed it springs to zero width, so the slot stays mounted for the portal
 */
export function FilePanelSlot() {
  const { slot, setSlot, ownerId, collapsed } = useFilePanelContext()
  const [sharedWidth, setSharedWidth] = useState(0)
  const [openWidth, setOpenWidth] = useState(0)
  const shown = ownerId !== null
  const expanded = shown && !collapsed

  useLayoutEffect(() => {
    const panel = slot?.parentElement
    const main = panel?.previousElementSibling
    if (!shown || !panel || !main) return
    const measureShared = () => Math.round(main.getBoundingClientRect().width + panel.getBoundingClientRect().width)
    const opened = measureShared()
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the open width is measured from the laid-out surfaces
    setSharedWidth(opened)
    setOpenWidth(Math.round(opened * OPEN_SHARE))
    // A window or left sidebar resize changes the shared width, the panel shrinks into the new limit
    const observer = new ResizeObserver(() => {
      const shared = measureShared()
      setSharedWidth(shared)
      setOpenWidth((width) => Math.min(width, Math.round(shared * MAX_SHARE)))
    })
    observer.observe(main)
    return () => observer.disconnect()
  }, [shown, slot])

  return (
    <SidePanelFrame
      width={expanded ? openWidth : 0}
      onWidthChange={setOpenWidth}
      maxWidth={Math.round(sharedWidth * MAX_SHARE)}
      className={cn("transition-[margin,background-color,box-shadow,border-color] duration-200",
        expanded ? "m-3 ml-0 rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]" : "pointer-events-none border-0")}
    >
      <div ref={setSlot} inert={!expanded} aria-hidden={!expanded} className="flex min-h-0 flex-1 flex-col" />
    </SidePanelFrame>
  )
}

/** Top bar toggle collapsing the file panel, only while it shows something. */
export function FilePanelToggle() {
  const { ownerId, collapsed, toggleCollapsed } = useFilePanelContext()
  if (ownerId === null) return null
  return (
    <button
      aria-label={collapsed ? "Open file panel" : "Close file panel"}
      onClick={toggleCollapsed}
      className="shrink-0 rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
    >
      {collapsed ? <PanelRight className="h-4 w-4" /> : <PanelRightClose className="h-4 w-4" />}
    </button>
  )
}

/** Show `children` in the tab's file panel under a header with `label`, `actions` and a close button. */
export function FilePanel({ label, onClose, actions, children }: {
  label: string
  onClose: () => void
  actions?: ReactNode
  children: ReactNode
}) {
  const { slot, ownerId, claim, release } = useFilePanelContext()
  const id = useId()
  const onCloseRef = useRef(onClose)
  useEffect(() => { onCloseRef.current = onClose }, [onClose])
  useEffect(() => {
    claim(id, () => onCloseRef.current())
    return () => release(id)
  }, [id, claim, release])

  if (ownerId !== id || !slot) return null
  return createPortal(
    <section aria-label={label} className="flex min-h-0 flex-1 flex-col">
      <div className="flex h-10 shrink-0 items-center gap-2 pl-4 pr-2 text-[11px] text-ink-subtle">
        <span className="min-w-0 flex-1 truncate">{label}</span>
        {actions}
        <button type="button" aria-label="Close panel" onClick={onClose} className="rounded-md p-1 text-ink-subtle transition-colors hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {children}
    </section>,
    slot,
  )
}

const LOAD_MESSAGES: Record<FileLoadFailure, string> = {
  forbidden: "This file is outside the folders gigachad-bot can read.",
  missing: "This file does not exist.",
  binary: "This is not a text file.",
  failed: "Could not load this file.",
}

/** Markdown renders as itself, any other text as a code block highlighted by its extension. */
function asMarkdown(path: string, content: string): string {
  return kindOf(path) === "markdown" ? content : fenceCode(extOf(path), content)
}

/** Image and PDF render from the raw file URL, the way the file finder previews them. */
function RawFileView({ path, kind }: { path: string; kind: "image" | "pdf" }) {
  if (kind === "pdf") return <div className="min-h-0 flex-1"><LazyPdfViewer key={path} url={fileViewerRawUrl(path)} /></div>
  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-4">
      {/* eslint-disable-next-line @next/next/no-img-element -- dynamic backend raw-file URL in a Tauri static export, next/image cannot optimize it */}
      <img src={fileViewerRawUrl(path)} alt={baseName(path)} className="mx-auto max-h-full max-w-full rounded border border-divider object-contain" />
    </div>
  )
}

/** Show `path` (absolute, inside a file root): image and PDF as such, text in Neovim (setting on) or read-only. */
function RootFileView({ path, onClose }: { path: string; onClose: () => void }) {
  const kind = kindOf(path)
  if (kind === "image" || kind === "pdf") return <FilePanel label={path} onClose={onClose}><RawFileView path={path} kind={kind} /></FilePanel>
  return <TextFileView path={path} onClose={onClose} />
}

/** Text file: in Neovim `:w` saves it in place and `:q` closes the panel. */
function TextFileView({ path, onClose }: { path: string; onClose: () => void }) {
  const neovim = useNeovimSetting()
  const write = useCallback((text: string) => writeRootFile(path, text), [path])
  const { content, loadError, reload, saveFromNeovim } = useRootFileDocument({ path, write, blankWhenMissing: false })
  useEffect(() => { reload() }, [reload])
  const neovimSession = useNeovimTerminal(neovim.enabled && content !== null ? { key: path, path, filename: baseName(path), content } : null, saveFromNeovim, onClose)

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
