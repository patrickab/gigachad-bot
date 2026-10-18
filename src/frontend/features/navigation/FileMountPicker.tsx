"use client"

import { useEffect, useEffectEvent, useRef, useState } from "react"
import { ChevronUp, Folder, FolderPlus } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { browseDirectories } from "@/lib/backend/api"
import type { DirectoryListing } from "@/lib/domain/types"

const HOME = "~"

/** Pick a backend directory to mount as a file mount.
 *
 * - Starts at the backend user's home, the list follows what is typed into the path
 * - Clicking a directory opens it, the up button opens its parent
 * - Enter or "Add" mounts the typed path
 */
export function FileMountPicker({ open, onClose, onConfirm }: {
  open: boolean
  onClose: () => void
  onConfirm: (path: string) => Promise<void>
}) {
  const [path, setPath] = useState(HOME)
  const [listing, setListing] = useState<DirectoryListing | null>(null)
  const [listingError, setListingError] = useState("")
  const [confirmError, setConfirmError] = useState("")
  const [confirming, setConfirming] = useState(false)
  const [wasOpen, setWasOpen] = useState(open)
  const inputRef = useRef<HTMLInputElement>(null)
  const requestRef = useRef(0)

  // Every opening starts again at home.
  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) {
      setPath(HOME)
      setListing(null)
      setConfirmError("")
    }
  }

  // Typing keeps its own spelling, opening a directory shows its resolved path. Only the latest request applies.
  const load = (target: string, showResolved: boolean) => {
    const request = ++requestRef.current
    return browseDirectories(target).then((result) => {
      if (request !== requestRef.current) return
      setListing(result)
      setListingError("")
      if (showResolved) setPath(result.path)
    }, (error: unknown) => {
      if (request === requestRef.current) setListingError(error instanceof Error ? error.message : "Cannot open this directory")
    })
  }

  const loadEvent = useEffectEvent(load)
  useEffect(() => {
    if (open) void loadEvent(HOME, true)
  }, [open])

  const openDirectory = (target: string) => {
    void load(target, true)
    inputRef.current?.focus()
  }

  const confirm = async () => {
    const target = path.trim()
    if (!target || confirming) return
    setConfirming(true)
    setConfirmError("")
    try {
      await onConfirm(target)
      onClose()
    } catch (error) {
      setConfirmError(error instanceof Error ? error.message : "Could not add this file mount")
    }
    setConfirming(false)
  }

  const childPath = (name: string) => listing!.path === "/" ? `/${name}` : `${listing!.path}/${name}`

  return (
    <Dialog open={open} onClose={onClose} icon={FolderPlus} title="Add file mount" initialFocusRef={inputRef} widthClassName="max-w-lg">
      <form onSubmit={(event) => { event.preventDefault(); void confirm() }} className="flex items-center gap-2 px-5 pb-3">
        <button
          type="button"
          aria-label="Open parent directory"
          disabled={!listing?.parent}
          onClick={() => listing?.parent && openDirectory(listing.parent)}
          className="shrink-0 rounded-lg border border-divider bg-surface/60 p-2 text-ink-subtle hover:text-ink disabled:opacity-30 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted transition-colors"
        >
          <ChevronUp className="h-4 w-4" />
        </button>
        <input
          ref={inputRef}
          aria-label="File mount path"
          value={path}
          onChange={(event) => { setPath(event.target.value); void load(event.target.value, false) }}
          autoComplete="off"
          spellCheck={false}
          className="min-w-0 flex-1 rounded-lg border border-divider bg-surface/60 px-3 py-2 font-mono text-xs text-ink placeholder-ink-faint outline-none transition-all duration-200 focus:border-ink-muted"
        />
        <button
          type="submit"
          disabled={confirming || !path.trim()}
          className="shrink-0 rounded-lg border border-divider bg-surface/60 px-3 py-2 text-xs text-ink-muted hover:text-ink disabled:opacity-30 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted transition-colors"
        >
          Add
        </button>
      </form>
      {(confirmError || listingError) && <p role="alert" className="px-5 pb-2 text-xs text-danger">{confirmError || listingError}</p>}
      <div role="list" aria-label="Directories" className="h-72 overflow-y-auto border-t border-divider/50 px-3 py-2">
        {listing?.directories.map((name) => (
          <div role="listitem" key={name}>
            <button
              type="button"
              onClick={() => openDirectory(childPath(name))}
              className="flex h-7 w-full min-w-0 items-center gap-2 rounded-md px-2 text-left font-mono text-xs text-ink-muted hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted transition-colors"
            >
              <Folder className="h-3.5 w-3.5 shrink-0 text-ink-faint" />
              <span className="truncate">{name}</span>
            </button>
          </div>
        ))}
        {listing?.directories.length === 0 && <p className="px-2 py-1 text-xs text-ink-faint">No subdirectories</p>}
      </div>
    </Dialog>
  )
}
