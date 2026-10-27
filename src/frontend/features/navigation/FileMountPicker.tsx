"use client"

import { useEffect, useEffectEvent, useRef, useState } from "react"
import { ChevronUp, Folder, FolderPlus } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { ErrorText } from "@/components/ui/ErrorText"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { browseDirectories } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"
import type { DirectoryListing } from "@/lib/domain/types"

const HOME = "~"
const REPOSITORY_LINK = /^(https:\/\/|ssh:\/\/|[\w.~-]+@[\w.-]+:)/

/** Whether *text* reads as an https or ssh git link, the backend decides if it is a valid one. */
const isRepositoryLink = (text: string) => REPOSITORY_LINK.test(text.trim())

/** Pick a backend directory to add as file mount.
 *
 * - Starts at the backend user's home, the list follows what is typed into the path
 * - Clicking a directory opens it, the up button opens its parent
 * - Enter or "Add" adds the typed path
 * - A pasted https or ssh git link is cloned instead, "Clone" replaces "Add"
 */
export function FileMountPicker({ open, onClose, onConfirm, onClone }: {
  open: boolean
  onClose: () => void
  onConfirm: (path: string) => Promise<void>
  onClone: (url: string) => Promise<void>
}) {
  const [path, setPath] = useState(HOME)
  const [listing, setListing] = useState<DirectoryListing | null>(null)
  const [listingError, setListingError] = useState("")
  const trimmed = path.trim()
  const { run: submit, pending: confirming, error: confirmError, setError: setConfirmError } = useAsyncAction(async () => {
    await (isRepositoryLink(trimmed) ? onClone(trimmed) : onConfirm(trimmed))
    onClose()
  }, isRepositoryLink(trimmed) ? "Could not clone this repository" : "Could not add this file mount")
  const inputRef = useRef<HTMLInputElement>(null)
  const requestRef = useRef(0)

  // Every opening starts again at home.
  useResetOnOpen(open, () => {
    setPath(HOME)
    setListing(null)
    setConfirmError("")
  })

  // Typing keeps its own spelling, opening a directory shows its resolved path. Only the latest request applies.
  const load = (target: string, showResolved: boolean) => {
    const request = ++requestRef.current
    return browseDirectories(target).then((result) => {
      if (request !== requestRef.current) return
      setListing(result)
      setListingError("")
      if (showResolved) setPath(result.path)
    }, (error: unknown) => {
      if (request === requestRef.current) setListingError(errorMessage(error, "Cannot open this directory"))
    })
  }

  const loadEvent = useEffectEvent(load)
  useEffect(() => {
    if (open) void loadEvent(HOME, true)
  }, [open])

  // A git link is no directory: drop the pending listing and its error instead of browsing it.
  const typeTarget = (target: string) => {
    if (!isRepositoryLink(target)) return void load(target, false)
    requestRef.current++
    setListingError("")
  }

  const openDirectory = (target: string) => {
    void load(target, true)
    inputRef.current?.focus()
  }

  const confirm = () => {
    if (trimmed && !confirming) void submit()
  }

  const childPath = (name: string) => listing!.path === "/" ? `/${name}` : `${listing!.path}/${name}`

  return (
    <Dialog open={open} onClose={onClose} icon={FolderPlus} title="Add file mount" initialFocusRef={inputRef} widthClassName="max-w-lg">
      <form onSubmit={(event) => { event.preventDefault(); confirm() }} className="flex items-center gap-2 px-5 pb-3">
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
          onChange={(event) => { setPath(event.target.value); typeTarget(event.target.value) }}
          autoComplete="off"
          spellCheck={false}
          className="min-w-0 flex-1 rounded-lg border border-divider bg-surface/60 px-3 py-2 font-mono text-xs text-ink placeholder-ink-faint outline-none transition-all duration-200 focus:border-ink-muted"
        />
        <button
          type="submit"
          disabled={confirming || !path.trim()}
          className="shrink-0 rounded-lg border border-divider bg-surface/60 px-3 py-2 text-xs text-ink-muted hover:text-ink disabled:opacity-30 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted transition-colors"
        >
          {isRepositoryLink(path) ? (confirming ? "Cloning…" : "Clone") : "Add"}
        </button>
      </form>
      <ErrorText className="px-5 pb-2">{confirmError || listingError}</ErrorText>
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
