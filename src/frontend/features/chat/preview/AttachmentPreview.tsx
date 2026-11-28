"use client"

import { useCallback, useState } from "react"
import { X } from "lucide-react"
import type { Attachment } from "@/lib/domain/types"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { useNeovimSetting, type NeovimSetting } from "@/contexts/SettingsContext"
import { FilePanel } from "@/features/editor/FilePanel"
import { ViewSwitch } from "@/components/ui/ViewSwitch"
import { NeovimEditor, useNeovimTerminal } from "@/features/editor/NeovimEditor"
import { PDF_MIME, RawAttachment, RenderedAttachment, isMarkdownAttachment, renderableMarkdown, type ContentChange } from "./AttachmentView"
import { useAttachmentLabel } from "@/features/chat/useAttachmentLabel"

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

function AttachmentPanel({ attachment, onClose, onContentChange }: {
  attachment: Attachment
  onClose: () => void
  onContentChange?: ContentChange
}) {
  const neovim = useNeovimSetting()
  const label = useAttachmentLabel(attachment)
  const views = viewsOf(attachment, neovim)
  const [chosenView, setView] = useState<PreviewView | null>(null)
  const view = chosenView && views.includes(chosenView) ? chosenView : views[0]
  // Neovim edits the attachment's own text, the rendered view shows it as markdown (code fenced by extension).
  const markdown = renderableMarkdown(attachment)
  const viewPicker = views.length > 1 && (
    <ViewSwitch value={view} onChange={setView} options={views.map((option) => [option, viewLabel(option, attachment)] as const)} />
  )

  return (
    <FilePanel label={label} onClose={onClose} actions={viewPicker}>
      {view === "raw" ? <RawAttachment attachment={attachment} />
        : view === "neovim" && attachment.content != null ? <NeovimAttachment attachment={attachment} content={attachment.content} onContentChange={onContentChange} onClose={onClose} />
          : view === "rendered" && markdown != null ? <RenderedAttachment attachment={attachment} markdown={markdown} onContentChange={isMarkdownAttachment(attachment) ? onContentChange : undefined} />
            : <div className="flex flex-1 items-center justify-center text-sm text-ink-subtle">No preview available</div>}
    </FilePanel>
  )
}

/** An image over the chat in a floating window sized to it: backdrop click, Escape or the close button close it. */
function ImageWindow({ attachment, onClose }: { attachment: Attachment; onClose: () => void }) {
  const label = useAttachmentLabel(attachment)
  return (
    <FloatingWindow label={label} onClose={onClose} panelClassName="relative h-auto max-h-[90vh] w-auto max-w-[90vw]">
      <button type="button" aria-label="Close image" onClick={onClose} className="absolute right-2 top-2 rounded-md bg-paper/80 p-1 text-ink-subtle transition-colors hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
        <X className="h-3.5 w-3.5" />
      </button>
      {/* eslint-disable-next-line @next/next/no-img-element -- attachment URLs are blob/data/backend URLs, not optimizable by next/image */}
      <img src={attachment.url} alt={attachment.name} className="block max-h-[90vh] max-w-[90vw] object-contain" />
    </FloatingWindow>
  )
}

/**
 * Show `attachment`, nothing while it is null: an image in a floating window, anything else in the tab's file panel.
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
  if (attachment.mime.startsWith("image/")) return <ImageWindow key={attachment.name} attachment={attachment} onClose={onClose} />
  return <AttachmentPanel key={attachment.name} attachment={attachment} onClose={onClose} onContentChange={onContentChange} />
}
