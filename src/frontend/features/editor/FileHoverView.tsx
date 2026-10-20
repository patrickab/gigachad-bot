"use client"

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react"
import { X } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { writeRootFile } from "@/lib/backend/api"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"
import { baseName, extOf, kindOf } from "@/lib/paths"
import { fenceCode } from "@gigachad/chat-ui"
import { NeovimEditor, useNeovimTerminal } from "./NeovimEditor"
import { useRootFileDocument, type FileLoadFailure } from "./useRootFileDocument"

type OpenFile = (path: string) => void

const FileHoverViewContext = createContext<OpenFile | null>(null)

/** Open a file in the floating file view, from anywhere under `FileHoverViewProvider`. */
export function useOpenFileHoverView(): OpenFile {
  const openFile = useContext(FileHoverViewContext)
  if (!openFile) throw new Error("useOpenFileHoverView must be used inside FileHoverViewProvider")
  return openFile
}

/** Own the one floating file view. Opening another file replaces the shown one. */
export function FileHoverViewProvider({ children }: { children: ReactNode }) {
  const [path, setPath] = useState<string | null>(null)
  const close = useCallback(() => setPath(null), [])
  return (
    <FileHoverViewContext.Provider value={setPath}>
      {children}
      {path && <FileHoverView key={path} path={path} onClose={close} />}
    </FileHoverViewContext.Provider>
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

/** Show `path` in a floating window: image and PDF as such, text in Neovim (setting on) or read-only. */
function FileHoverView({ path, onClose }: { path: string; onClose: () => void }) {
  const kind = kindOf(path)
  const filename = baseName(path)
  const header = (
    <div className="flex h-9 shrink-0 items-center gap-2 border-b border-divider/30 pl-4 pr-2 text-[11px] text-ink-subtle">
      <span className="min-w-0 flex-1 truncate">{path}</span>
      <button type="button" aria-label="Close file" onClick={onClose} className="rounded p-1 text-ink-subtle transition-colors hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  )
  if (kind === "image" || kind === "pdf") {
    return <FloatingWindow onClose={onClose} label={filename}>{header}<RawFileView path={path} kind={kind} /></FloatingWindow>
  }
  return <TextFileHoverView path={path} filename={filename} header={header} onClose={onClose} />
}

/**
 * Text file body of the floating view (absolute path, inside a file root).
 *
 * - Neovim editor setting on: the file in Neovim, `:w` saves it in place, `:q` closes the window. A click outside does not close it, that would discard unsaved edits
 * - Off: read-only rendered view
 */
function TextFileHoverView({ path, filename, header, onClose }: { path: string; filename: string; header: ReactNode; onClose: () => void }) {
  const neovim = useNeovimSetting()
  const write = useCallback((text: string) => writeRootFile(path, text), [path])
  const { content, loadError, reload, saveFromNeovim } = useRootFileDocument({ path, write, blankWhenMissing: false })
  useEffect(() => { reload() }, [reload])
  const neovimSession = useNeovimTerminal(neovim.enabled && content !== null ? { key: path, path, filename, content } : null, saveFromNeovim, onClose)

  return (
    <FloatingWindow onClose={onClose} label={filename} dismissOnBackdrop={!neovim.enabled}>
      {header}
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
    </FloatingWindow>
  )
}
