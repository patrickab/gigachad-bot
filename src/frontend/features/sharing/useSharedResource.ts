"use client"

import { useEffect, useState } from "react"
import { readSharedResource, type SharedResource } from "@/lib/backend/api"

/** What a share link opens: undefined while loading, null for an unknown, rotated or revoked link. */
export function useSharedResource(secret: string): SharedResource | null | undefined {
  const [resource, setResource] = useState<SharedResource | null | undefined>(undefined)
  useEffect(() => {
    let live = true
    readSharedResource(secret).then(
      (opened) => { if (live) setResource(opened) },
      () => { if (live) setResource(null) },
    )
    return () => { live = false }
  }, [secret])
  return resource
}
