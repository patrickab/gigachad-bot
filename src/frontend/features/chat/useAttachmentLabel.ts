"use client"

import { useDisplayName } from "@/contexts/ProjectContext"
import { useTabValue } from "@/contexts/TabContext"
import type { Attachment } from "@/lib/domain/types"

/** The label of a chat attachment: the display name the tab's project gives a referenced file, else the attachment's name. */
export function useAttachmentLabel(attachment: Attachment): string {
  const slug = useTabValue()?.tab.projectSlug
  const shownName = useDisplayName()
  return (attachment.vaultPath && shownName(slug, attachment.vaultPath)) || attachment.name
}
