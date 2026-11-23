"use client"

import { useEffect, useState } from "react"
import { Copy, Link2 } from "lucide-react"
import { IconButton } from "@/components/ui/IconButton"
import { MenuPopover } from "@/components/ui/MenuPopover"
import { ErrorText } from "@/components/ui/ErrorText"
import { createShare, readShareAccess, revokeShare, shareLinkUrl, type ShareAccess, type ShareTarget } from "@/lib/backend/api"
import { cn } from "@/lib/utils"

/** A live link. Its URL is known only right after creating it: the backend keeps just a hash of the secret. */
type Link = { access: ShareAccess; url?: string }

const ACTION_CLASS = "rounded-md px-2 py-1 text-ink-subtle transition-colors hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** Opens the share panel of one resource: create, copy, replace or revoke its link. `className` matches the trigger to its neighbors. */
export function ShareButton({ target, className }: { target: ShareTarget; className?: string }) {
  return (
    <MenuPopover trigger={(open, toggle) => <IconButton icon={Link2} label="Share" onClick={toggle} className={cn(className, open && "bg-hover text-ink")} />}>
      {(open) => open && <SharePanel target={target} />}
    </MenuPopover>
  )
}

function SharePanel({ target }: { target: ShareTarget }) {
  const { kind, key } = target
  // undefined while loading, null without a link
  const [link, setLink] = useState<Link | null | undefined>(undefined)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let live = true
    readShareAccess({ kind, key }).then(
      (access) => { if (live) setLink(access && { access }) },
      () => { if (live) setFailed(true) },
    )
    return () => { live = false }
  }, [kind, key])

  const run = (action: () => Promise<Link | null>) => {
    setFailed(false)
    action().then(setLink, () => setFailed(true))
  }
  const create = () => run(async () => {
    const { secret, access } = await createShare(target)
    const url = shareLinkUrl(secret)
    await copy(url)
    return { access, url }
  })
  const revoke = () => run(async () => {
    await revokeShare(target)
    return null
  })

  return (
    <div role="dialog" aria-label="Share" className="absolute right-0 top-full z-20 mt-1 flex w-72 flex-col gap-2 rounded-xl bg-surface-elevated p-2 text-xs shadow-[var(--shadow-md)]">
      {link === null && <button type="button" onClick={create} className={cn(ACTION_CLASS, "self-start")}>Create link</button>}
      {link && (
        <>
          <span className="px-2 text-ink-faint">Anyone with the link can {link.access}</span>
          {link.url && (
            <div className="flex items-center gap-1">
              <input readOnly aria-label="Share link" value={link.url} onFocus={(event) => event.target.select()} className="min-w-0 flex-1 rounded-md bg-paper px-2 py-1 text-ink outline-none" />
              <IconButton icon={Copy} label="Copy link" onClick={() => void copy(link.url!)} />
            </div>
          )}
          <div className="flex justify-end gap-1">
            {!link.url && <button type="button" onClick={create} className={ACTION_CLASS}>New link</button>}
            <button type="button" onClick={revoke} className={cn(ACTION_CLASS, "hover:text-danger")}>Revoke</button>
          </div>
        </>
      )}
      <ErrorText className="px-2">{failed && "Sharing failed. Try again."}</ErrorText>
    </div>
  )
}

/** Best effort: the URL stays selectable in the panel when the clipboard is unavailable. */
async function copy(url: string): Promise<void> {
  await navigator.clipboard?.writeText(url).catch(() => {})
}
