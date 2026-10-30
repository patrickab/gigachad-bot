"use client"

import { useEffect, useRef, useState, type ReactNode } from "react"
import { Eye, EyeOff, File as FileIcon, FileX, X } from "lucide-react"
import type { Attachment } from "@/lib/domain/types"
import { cn } from "@/lib/utils"
import { AttachmentIcon } from "@/features/chat/preview/AttachmentView"
import { useAttachmentHoverCard } from "@/features/chat/preview/AttachmentHoverCard"
import { useAttachmentLabel } from "@/features/chat/useAttachmentLabel"

/** Attachment of an earlier message, still part of the chat's context unless switched inactive. */
export interface ContextAttachment {
  messageIndex: number
  attachment: Attachment
}

const CHIP_BUTTON_CLASS = "shrink-0 rounded p-0.5 transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/**
 * One attachment pill above the composer.
 *
 * - Name opens the full preview, hovering it peeks first
 * - Icon switches the attachment in and out of context, when `onToggleActive` is given; inactive chips are faded
 * - `uploading` shows a spinner in place of the icon, a reference whose file is gone (`missing`) a red crossed-out file
 */
export function AttachmentChip({ attachment, uploading, onOpen, onRemove, onToggleActive }: {
  attachment: Attachment
  uploading?: boolean
  onOpen: () => void
  onRemove: () => void
  onToggleActive?: () => void
}) {
  const hover = useAttachmentHoverCard(attachment)
  const label = useAttachmentLabel(attachment)
  const icon = uploading
    ? <span className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-ink-faint border-t-ink" />
    : attachment.missing
      ? <FileX className="h-3.5 w-3.5 shrink-0 text-danger" aria-hidden="true" />
      : <AttachmentIcon mime={attachment.mime} generic={FileIcon} size="h-3.5 w-3.5" genericMuted={false} />
  const ToggleIcon = attachment.active ? EyeOff : Eye

  return (
    <div className={cn(
      "group/chip inline-flex max-w-56 items-center gap-1 rounded-xl border border-divider-strong/50 bg-surface-elevated py-1 pl-1.5 pr-1 text-xs transition-opacity",
      !attachment.active && "opacity-50 hover:opacity-80 focus-within:opacity-80",
    )}>
      {onToggleActive ? (
        <button
          type="button"
          aria-label={attachment.active ? `Exclude ${label} from context` : `Include ${label} in context`}
          aria-pressed={attachment.active}
          onClick={onToggleActive}
          className={cn(CHIP_BUTTON_CLASS, "group/toggle text-ink-muted hover:text-ink")}
        >
          <span className="group-hover/toggle:hidden group-focus-visible/toggle:hidden">{icon}</span>
          <ToggleIcon className="hidden h-3.5 w-3.5 group-hover/toggle:block group-focus-visible/toggle:block" />
        </button>
      ) : <span className="p-0.5">{icon}</span>}
      <button
        type="button"
        aria-label={attachment.missing ? `Preview ${label}, file not found, last known version` : `Preview ${label}`}
        onClick={onOpen}
        {...hover.anchorProps}
        className="min-w-0 truncate rounded text-ink hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
      >
        {label}
      </button>
      <button
        type="button"
        aria-label={`Remove ${label}`}
        onClick={onRemove}
        className={cn(CHIP_BUTTON_CLASS, "text-ink-faint opacity-0 hover:text-danger focus-visible:opacity-100 group-hover/chip:opacity-100")}
      >
        <X className="h-3 w-3" />
      </button>
      {hover.card}
    </div>
  )
}

/** Grace period to move the pointer from the "+N" pill into its popover. */
const POPOVER_HIDE_DELAY_MS = 150

/**
 * "+N" pill that gathers the attachments out of context; hovering or focusing it opens a popover of their chips.
 *
 * - Hovering a chip in the popover peeks at it like any chip, the popover stays open while the pointer is on that card
 * - `children` are the chips
 */
export function InactiveAttachments({ count, children }: { count: number; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const hideTimer = useRef<number | undefined>(undefined)
  const show = () => { window.clearTimeout(hideTimer.current); setOpen(true) }
  const hideSoon = () => { window.clearTimeout(hideTimer.current); hideTimer.current = window.setTimeout(() => setOpen(false), POPOVER_HIDE_DELAY_MS) }
  useEffect(() => () => window.clearTimeout(hideTimer.current), [])

  return (
    <div
      className="relative"
      onMouseEnter={show}
      onMouseLeave={hideSoon}
      onFocus={show}
      onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false) }}
      onKeyDown={(event) => { if (event.key === "Escape") setOpen(false) }}
    >
      <button
        type="button"
        aria-label={`${count} attachments out of context`}
        aria-expanded={open}
        className="inline-flex items-center rounded-xl border border-divider-strong/50 bg-surface-elevated px-2 py-1 text-xs text-ink-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
      >
        +{count}
      </button>
      {open && (
        <div className="absolute bottom-full left-0 pb-2">
          <div className="flex w-max max-w-xs flex-col items-start gap-1.5 rounded-xl border border-divider bg-paper p-2 shadow-[var(--shadow-xl)]">
            {children}
          </div>
        </div>
      )}
    </div>
  )
}
