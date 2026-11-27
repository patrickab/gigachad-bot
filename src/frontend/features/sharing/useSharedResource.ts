"use client"

import { useLoaded } from "@/hooks/useLoaded"
import { readSharedResource, type SharedResource } from "@/lib/backend/api"

/** What a share link opens: undefined while loading, null for an unknown, rotated or revoked link. */
export function useSharedResource(secret: string): SharedResource | null | undefined {
  const { value, error } = useLoaded(secret, readSharedResource)
  return error ? null : value
}
