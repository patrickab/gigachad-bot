"use client"

import { useCallback, useState } from "react"
import { createPortal } from "react-dom"
import { X } from "lucide-react"
import type { Attachment } from "@/lib/domain/types"
import { cn } from "@/lib/utils"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { useNeovimSetting, type NeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, useNeovimTerminal } from "@/features/editor/NeovimEditor"
import { PDF_MIME, RawAttachment, RenderedAttachment, isMarkdownAttachment, renderableMarkdown, type ContentChange } from "./AttachmentView"

type PreviewView = "raw" | "rendered" | "neovim"

/**
 * Ways to show `attachment`, the first being the default. Empty when there is nothing to show.
 *
 * - Neovim is offered whenever the backend runs it, the device's Neovim preference only makes it the default
 */
function viewsOf(attachment: Attachment, neovim: NeovimSetting): PreviewView[] {
  if (attachment.mime.startsWith("image/")) return ["raw"]
  if (attachment.mime === PDF_MIME) return attachment.parsedMd ? ["raw", "rendered"] : ["raw"]
  if (attachment.content == null) return []
  if (!neovim.available) return ["rendered"]
  return neovim.enabled ? ["neovim", "rendered"] : ["rendered", "neovim"]
}

function viewLabel(view: PreviewView, attachment: Attachment): string {
  if (view === "raw") return attachment.mime === PDF_MIME ? "PDF" : "Image"
  return view === "neovim" ? "Neovim" : "Rendered"
}

/** Text attachment in Neovim: `:w` stores the buffer through `onContentChange` (read-only without it), `:q` closes. */
function NeovimAttachment({ attachment, content, onContentChange, onClose }: {
  attachment: Attachment
  content: string
  onContentChange?: ContentChange
  onClose: () => void
}) {
  const save = useCallback(async (text: string) => {
    if (!onContentChange) return content
    await onContentChange(text)
    return text
  }, [content, onContentChange])
  const session = useNeovimTerminal({ key: attachment.name, path: attachment.name, filename: attachment.name, content }, save, onClose)
  return <NeovimEditor terminal={session.terminal} status={session.status} onRestart={session.restart} />
}

function PreviewWindow({ attachment, onClose, onContentChange }: {
  attachment: Attachment
  onClose: () => void
  onContentChange?: ContentChange
}) {
  const neovim = useNeovimSetting()
  const views = viewsOf(attachment, neovim)
  const [chosenView, setView] = useState<PreviewView | null>(null)
  const view = chosenView && views.includes(chosenView) ? chosenView : views[0]
  // Neovim edits the attachment's own text, the rendered view shows it as markdown (code fenced by extension).
  const markdown = renderableMarkdown(attachment)

  return (
    // A click outside would discard unsaved Neovim edits, so only the other views close on it.
    <FloatingWindow onClose={onClose} label={attachment.name} panelClassName="max-w-4xl" dismissOnBackdrop={view !== "neovim"}>
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-divider/30 pl-4 pr-2 text-[11px] text-ink-subtle">
        <span className="min-w-0 flex-1 truncate">{attachment.name}</span>
        {views.length > 1 && (
          <div role="group" aria-label="Show as" className="flex gap-px rounded-md bg-surface p-0.5">
            {views.map((option) => (
              <button key={option} type="button" aria-pressed={view === option} onClick={() => setView(option)} className={cn("rounded px-2 py-0.5 text-xs transition-colors", view === option ? "bg-surface-elevated text-ink" : "text-ink-subtle hover:text-ink")}>
                {viewLabel(option, attachment)}
              </button>
            ))}
          </div>
        )}
        <button type="button" aria-label="Close attachment preview" onClick={onClose} className="rounded p-1 text-ink-subtle transition-colors hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {view === "raw" ? <RawAttachment attachment={attachment} />
        : view === "neovim" && attachment.content != null ? <NeovimAttachment attachment={attachment} content={attachment.content} onContentChange={onContentChange} onClose={onClose} />
          : view === "rendered" && markdown != null ? <RenderedAttachment attachment={attachment} markdown={markdown} onContentChange={isMarkdownAttachment(attachment) ? onContentChange : undefined} />
            : <div className="flex flex-1 items-center justify-center text-sm text-ink-subtle">No preview available</div>}
    </FloatingWindow>
  )
}

/**
 * Floating window previewing `attachment`, nothing while it is null.
 *
 * - PDFs switch between the PDF and its rendered extraction, when MinerU produced one
 * - Text attachments switch between Neovim (setting on, the default then) and rendered markdown
 * - `onContentChange` stores edits of text attachments: Neovim `:w`, markdown task-list checkboxes
 */
export function AttachmentPreview({ attachment, onClose, onContentChange }: {
  attachment: Attachment | null
  onClose: () => void
  onContentChange?: ContentChange
}) {
  if (!attachment) return null
  // Portaled: the composer animates with a transform, which would trap a fixed overlay inside it.
  return createPortal(
    <PreviewWindow key={attachment.name} attachment={attachment} onClose={onClose} onContentChange={onContentChange} />,
    document.body,
  )
}
