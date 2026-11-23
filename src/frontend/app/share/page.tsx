"use client"

import { useEffect, useSyncExternalStore } from "react"
import { LazySharedCanvas } from "@/features/editor/canvas/LazyCanvas"
import { useSharedResource } from "@/features/sharing/useSharedResource"
import { sharedContentUrl } from "@/lib/backend/api"
import { LoadingPlaceholder } from "@/components/ui/DocumentLoadState"

const subscribeHash = (onChange: () => void) => {
  window.addEventListener("hashchange", onChange)
  return () => window.removeEventListener("hashchange", onChange)
}

/** A share link, `/share#<secret>`: the one resource it grants, full page. */
export default function SharePage() {
  const secret = useSyncExternalStore(subscribeHash, () => window.location.hash.slice(1), () => null)
  if (secret === null) return null
  return (
    <main className="flex h-dvh flex-col bg-paper">
      <SharedResourceView key={secret} secret={secret} />
    </main>
  )
}

function SharedResourceView({ secret }: { secret: string }) {
  const resource = useSharedResource(secret)
  useEffect(() => {
    if (resource) document.title = resource.name
  }, [resource])
  if (resource === undefined) return <LoadingPlaceholder size="fill" />
  if (resource === null) return <p className="m-auto text-sm text-ink-faint">This link is invalid or was revoked</p>
  if (resource.view === "canvas") return <LazySharedCanvas secret={secret} />
  // The browser renders the bytes itself (images, PDFs, text), sandboxed by the response's own headers.
  return <iframe title={resource.name} src={sharedContentUrl(secret)} className="min-h-0 flex-1 border-0" />
}
