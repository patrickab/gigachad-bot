"use client"

import { CollaborativeCanvasEditor } from "./CanvasEditor"
import { sharedCanvasEndpoint } from "@/lib/backend/api"

/** A canvas opened through a share link: the same collaboration as the owner's, through the link's routes. */
export function SharedCanvas({ secret }: { secret: string }) {
  return <CollaborativeCanvasEditor endpoint={sharedCanvasEndpoint(secret)} />
}
