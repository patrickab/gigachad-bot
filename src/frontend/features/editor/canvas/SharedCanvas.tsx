"use client"

import { CanvasEditor } from "./CanvasEditor"
import { useCollaborativeCanvas } from "./useCollaborativeCanvas"
import { sharedCanvasEndpoint } from "@/lib/backend/api"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"

/** A canvas opened through a share link: the same collaboration as the owner's, through the link's routes. */
export function SharedCanvas({ secret }: { secret: string }) {
  const { document, replace, status, retry } = useCollaborativeCanvas(sharedCanvasEndpoint(secret))
  if (!document) {
    return status === "error"
      ? <LoadFailure message="Failed to load this canvas." onRetry={retry} size="fill" />
      : <LoadingPlaceholder size="fill" />
  }
  return <CanvasEditor doc={document} onChange={replace} />
}
