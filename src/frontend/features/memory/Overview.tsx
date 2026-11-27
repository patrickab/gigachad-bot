"use client"

import type { ReactNode } from "react"
import { FileText, Folder, List } from "lucide-react"
import type { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { formatUpdated, type MemoryLocation } from "@/features/memory/memoryUtils"
import { relativeIn } from "@/lib/paths"
import type { MemorySection } from "@/features/memory/useMemoryLibrary"

/** The listed level: at the root a section per memory_store, else the one memory_store or directory. Rows open nested levels, memories and documents. */
export function Overview({ sections, showHeadings, onOpenLocation, onOpenPage }: {
  sections: MemorySection[]
  showHeadings: boolean
  onOpenLocation: (location: MemoryLocation) => void
  onOpenPage: (memoryStore: string, path: string) => void
}) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-6">
      {sections.map((section) => (
        <section key={`${section.memoryStore}:${section.path}`} aria-label={section.memoryStore}>
          {showHeadings && (
            <button onClick={() => onOpenLocation({ memoryStore: section.memoryStore, path: "" })}
              className="mx-1 mb-1 mt-4 rounded-md px-2 py-0.5 text-sm font-semibold text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
              {section.memoryStore}
            </button>
          )}
          <SectionRows section={section} onOpenLocation={onOpenLocation} onOpenPage={onOpenPage} />
        </section>
      ))}
    </div>
  )
}

function SectionRows({ section, onOpenLocation, onOpenPage }: {
  section: MemorySection
  onOpenLocation: (location: MemoryLocation) => void
  onOpenPage: (memoryStore: string, path: string) => void
}) {
  const { memoryStore, path, listing } = section
  const empty = !listing.memory_stores.length && !listing.directories.length && !listing.memories.length && !listing.documents.length
  if (empty) return <p className="px-3 py-2.5 text-sm text-ink-faint">Empty</p>

  return (
    <>
      {listing.memory_stores.map((nested) => (
        <Row key={`store:${nested}`} icon={Folder} name={relativeIn(memoryStore, nested)}
          onOpen={() => onOpenLocation({ memoryStore: nested, path: "" })} />
      ))}
      {listing.directories.map((directory) => (
        <Row key={`dir:${directory}`} icon={Folder} name={relativeIn(path, directory)}
          onOpen={() => onOpenLocation({ memoryStore, path: directory })} />
      ))}
      {listing.memories.map((memory) => (
        <Row key={`memory:${memory.path}`} icon={List} name={relativeIn(path, memory.path)}
          secondary={memory.error ? <span className="text-danger">Unreadable</span> : memory.description}
          updated={formatUpdated(memory.updated)}
          onOpen={() => onOpenPage(memoryStore, memory.path)} />
      ))}
      {listing.documents.map((document) => (
        <Row key={`doc:${document.path}`} icon={FileText} name={relativeIn(path, document.path)}
          updated={formatUpdated(document.modified)}
          onOpen={() => onOpenPage(memoryStore, document.path)} />
      ))}
    </>
  )
}

function Row({ icon: Icon, name, secondary, updated, onOpen }: {
  icon: LucideIcon
  name: string
  secondary?: ReactNode
  updated?: string
  onOpen: () => void
}) {
  return (
    <button onClick={onOpen} aria-label={`Open ${name}`}
      className={cn("grid w-full min-w-0 grid-cols-[minmax(0,14rem)_minmax(0,1fr)_9rem] items-center gap-4 rounded-lg border-b border-divider/40 px-3 py-2.5 text-left text-sm transition-colors",
        "hover:border-transparent hover:bg-hover focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted")}>
      <span className="flex min-w-0 items-center gap-2 text-ink">
        <Icon className="h-3.5 w-3.5 shrink-0 text-ink-subtle" aria-hidden="true" />
        <span className="truncate">{name}</span>
      </span>
      <span className="truncate text-ink-muted">{secondary}</span>
      <span className="truncate text-ink-subtle">{updated}</span>
    </button>
  )
}
