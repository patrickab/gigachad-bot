"use client"

import { Plus } from "lucide-react"
import { useState } from "react"
import { CommandPicker } from "@/components/ui/CommandPicker"
import { latexPreviewUrl } from "@/lib/backend/urls"
import type { LatexPrimitive } from "@/lib/domain/latex"

/** The primitives as one headed group per primitive group, groups and primitives in the order given. */
function groupsOf(primitives: LatexPrimitive[]) {
  const groups = new Map<string, { key: string; label: string; items: LatexPrimitive[] }>()
  for (const primitive of primitives) {
    const group = groups.get(primitive.group) ?? { key: primitive.group, label: primitive.group_label, items: [] }
    group.items.push(primitive)
    groups.set(primitive.group, group)
  }
  return [...groups.values()]
}

/** The top bar's "Add slide" button: a filterable list of the presentation's slide primitives by group, the highlighted one rendered in the preview window, the picked one goes to `onInsert`, the list stays open when it returns false. */
export function AddSlide({ theme, primitives, onInsert }: { theme: string; primitives: LatexPrimitive[]; onInsert: (text: string) => boolean }) {
  // Previews that failed to load show no window
  const [failed, setFailed] = useState<ReadonlySet<string>>(new Set())
  return <CommandPicker<LatexPrimitive>
    triggerLabel="Add slide"
    triggerContent={<>
      <Plus className="h-3.5 w-3.5 shrink-0" />
      <span className="truncate">Add slide</span>
    </>}
    triggerClassName="px-2.5"
    label="Slide primitives"
    groups={groupsOf(primitives)}
    itemKey={(primitive) => primitive.id}
    searchPlaceholder="Search slides…"
    searchText={(primitive) => `${primitive.label} ${primitive.group} ${primitive.summary}`}
    renderItem={(primitive) => <span className="flex min-w-0 flex-1 items-baseline gap-2">
      <span className="shrink-0 leading-5">{primitive.label}</span>
      <span className="truncate text-[10px] text-ink-faint">{primitive.summary}</span>
    </span>}
    renderPreview={(primitive) => {
      if (primitive.preview === null) return null
      const src = latexPreviewUrl(theme, primitive.id, primitive.preview)
      if (failed.has(src)) return null
      // eslint-disable-next-line @next/next/no-img-element -- dynamic backend preview URL in a Tauri static export, next/image cannot optimize it
      return <img key={src} src={src} alt="" draggable={false} onError={() => setFailed((current) => new Set(current).add(src))} className="block h-full w-full object-contain" />
    }}
    previewAspect={16 / 9}
    onPick={(primitive) => onInsert(primitive.text)}
    placement="below"
    panelClassName="w-72"
  />
}
