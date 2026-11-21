"use client"

import { useEffect, useEffectEvent, useRef, useState, type KeyboardEvent } from "react"
import { ChevronUp, Folder, FolderPlus } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { ErrorText } from "@/components/ui/ErrorText"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import { browseDirectories } from "@/lib/backend/api"
import { fuzzyScore, rankByScore } from "@/lib/fuzzy"
import { cn, errorMessage } from "@/lib/utils"
import type { DirectoryListing } from "@/lib/domain/files"

const HOME = "~"

/** *path* with one trailing `/`, so the next typed segment filters its folders. */
const withSlash = (path: string) => path.endsWith("/") ? path : `${path}/`
const REPOSITORY_LINK = /^(https:\/\/|ssh:\/\/|[\w.~-]+@[\w.-]+:)/

/** Whether *text* reads as an https or ssh git link, the backend decides if it is a valid one. */
const isRepositoryLink = (text: string) => REPOSITORY_LINK.test(text.trim())

/** The directory a typed path lists and the fragment after its last `/` that filters it: `~/git/gi` is `~/git/` and `gi`. */
export function splitPath(text: string): { dir: string; fragment: string } {
  if (text === HOME) return { dir: HOME, fragment: "" }
  const slash = text.lastIndexOf("/")
  if (slash === -1) return { dir: HOME, fragment: text }
  return { dir: text.slice(0, slash + 1), fragment: text.slice(slash + 1) }
}

/** Subdirectories matching *fragment* best first; with none typed, visible folders before dot folders. */
function matches(names: readonly string[], fragment: string): string[] {
  if (fragment) return rankByScore(names, (name) => fuzzyScore(name, fragment))
  return [...names.filter((name) => !name.startsWith(".")), ...names.filter((name) => name.startsWith("."))]
}

/** Pick a backend directory to add as file mount.
 *
 * - Starts at the backend user's home, the list shows the folders of the typed path's last directory, fuzzy filtered by what follows its last `/`
 * - Up/Down move the highlight, Tab completes it into the path and lists its folders, Enter completes it too while the typed fragment is no folder yet
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
  const [active, setActive] = useState(0)
  // A trailing `/` only leads the list on, the mounted path has none
  const trimmed = path.trim().replace(/(.)\/+$/, "$1")
  const { dir, fragment } = splitPath(path)
  const shown = matches(listing?.directories ?? [], fragment)
  const highlighted = shown[Math.min(active, shown.length - 1)] as string | undefined
  const listRef = useRef<HTMLDivElement>(null)
  const browsedRef = useRef(HOME)
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
    browsedRef.current = target
    return browseDirectories(target).then((result) => {
      if (request !== requestRef.current) return
      setListing(result)
      setListingError("")
      if (showResolved) setPath(withSlash(result.path))
    }, (error: unknown) => {
      if (request === requestRef.current) setListingError(errorMessage(error, "Cannot open this directory"))
    })
  }

  const loadEvent = useEffectEvent(load)
  useEffect(() => {
    if (open) void loadEvent(HOME, true)
  }, [open])

  // A git link is no directory: drop the pending listing and its error instead of browsing it.
  const typeTarget = (text: string) => {
    setActive(0)
    if (isRepositoryLink(text)) {
      requestRef.current++
      return setListingError("")
    }
    // Typing within one directory only filters its list
    const { dir: target } = splitPath(text)
    if (target !== browsedRef.current) void load(target, false)
  }

  /** Put *name* of the listed directory into the path and list it. */
  const complete = (name: string) => {
    const next = `${dir}${name}/`
    setPath(next)
    setActive(0)
    void load(next, false)
    inputRef.current?.focus()
  }

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (shown.length === 0) return
      event.preventDefault()
      setActive((current) => (Math.min(current, shown.length - 1) + (event.key === "ArrowDown" ? 1 : -1) + shown.length) % shown.length)
    } else if (event.key === "Tab" && !event.shiftKey && highlighted !== undefined && !isRepositoryLink(path)) {
      event.preventDefault()
      complete(highlighted)
    } else if (event.key === "Enter" && fragment && highlighted !== undefined && !shown.includes(fragment) && !isRepositoryLink(path)) {
      // The fragment names no folder yet: Enter picks the best match instead of adding a path that does not exist
      event.preventDefault()
      complete(highlighted)
    }
  }

  const openDirectory = (target: string) => {
    void load(target, true)
    inputRef.current?.focus()
  }

  const confirm = () => {
    if (trimmed && !confirming) void submit()
  }

  const childPath = (name: string) => listing!.path === "/" ? `/${name}` : `${listing!.path}/${name}`

  // The highlight stays in view while arrowing
  useEffect(() => {
    listRef.current?.querySelector("[data-active]")?.scrollIntoView?.({ block: "nearest" })
  }, [active, shown.length])

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
          onKeyDown={onKeyDown}
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
      <div ref={listRef} role="list" aria-label="Directories" className="h-72 overflow-y-auto border-t border-divider/50 px-2 py-2">
        {shown.map((name) => (
          <div role="listitem" key={name}>
            <button
              type="button"
              tabIndex={-1}
              data-active={name === highlighted || undefined}
              onClick={() => openDirectory(childPath(name))}
              onMouseMove={() => setActive(shown.indexOf(name))}
              className={cn(
                "flex h-8 w-full min-w-0 items-center gap-2.5 rounded-lg px-2.5 text-left text-[13px] text-ink-muted transition-colors duration-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted",
                name === highlighted && "bg-surface-elevated text-ink",
              )}
            >
              <Folder className="h-3.5 w-3.5 shrink-0 text-ink-faint" />
              <span className={cn("truncate", name.startsWith(".") && "text-ink-faint")}>{name}</span>
            </button>
          </div>
        ))}
        {listing && shown.length === 0 && <p className="px-2.5 py-1 text-xs text-ink-faint">{fragment ? "No matching folders" : "No subdirectories"}</p>}
      </div>
    </Dialog>
  )
}
