"use client"

import { X } from "lucide-react"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { IconButton } from "@/components/ui/IconButton"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { childLabel, locationLabel } from "@/features/memory/memoryUtils"
import { useMemoryViewer } from "@/features/memory/MemoryViewerContext"
import { PageHeader, Spinner } from "@/features/memory/MemoryChrome"
import { DocumentPage, MemoryPage } from "@/features/memory/MemoryPage"
import { Overview } from "@/features/memory/Overview"
import { useMemoryLibrary } from "@/features/memory/useMemoryLibrary"

export function MemoryViewer() {
  const { isOpen, closeMemoryViewer } = useMemoryViewer()
  return isOpen ? <MemoryWindow onClose={closeMemoryViewer} /> : null
}

/** Browser over the memory root: memory_stores, directories, memories and documents. Esc goes back one level, at the root it closes. */
function MemoryWindow({ onClose }: { onClose: () => void }) {
  const { location, trail, sections, target, page, error, setError, openLocation, openPage, back, add, update, retire, describe } = useMemoryLibrary()
  const parent = target ? location : trail.at(-2)
  useEscapeLayer(back, Boolean(parent))

  let content
  if (target) {
    content = !page ? (!error ? <Spinner label="Loading memory" /> : null)
      : page.kind === "memory" ? (
        <MemoryPage key={`${target.memoryStore}:${target.path}`} description={page.description} facts={page.facts}
          onDescribe={describe} onAdd={add} onUpdate={update} onRetire={retire} />
      ) : <DocumentPage text={page.text} />
  } else {
    content = sections
      ? <Overview sections={sections} showHeadings={!location.memoryStore} onOpenLocation={openLocation} onOpenPage={openPage} />
      : !error ? <Spinner label="Loading memory" /> : null
  }

  return (
    <FloatingWindow onClose={onClose} label="Memory" overlayClassName="z-[95]"
      panelClassName="h-auto max-h-[min(86vh,840px)] max-w-4xl bg-surface">
      <div className="flex shrink-0 items-center justify-between gap-4 px-6 pb-2 pt-5">
        {parent
          ? <PageHeader parent={locationLabel(parent)} current={childLabel(parent, target ?? location)} onBack={back} />
          : <h2 className="text-base font-semibold text-ink">Memory</h2>}
        <IconButton icon={X} label="Close memory" onClick={onClose} />
      </div>
      {content}
      {error && (
        <div role="alert" className="flex shrink-0 items-center justify-between gap-3 border-t border-divider/50 px-6 py-2 text-xs text-danger">
          <span>{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss memory error" className="text-ink-muted hover:text-ink">
            <X className="h-3 w-3" />
          </button>
        </div>
      )}
    </FloatingWindow>
  )
}
