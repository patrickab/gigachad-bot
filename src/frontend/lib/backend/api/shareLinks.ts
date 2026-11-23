// Share links (`backend/routes/share_links.py`): the owner's `/shares` routes and a secret's `/shared/<secret>` routes.
import { getApiBase } from "../config"
import { toQuery } from "../urls"
import type { CanvasEndpoint } from "./canvases"
import { del, post, request } from "./http"

export type ShareAccess = "view" | "edit"

/** One shareable resource of the owner: a document key or an asset path. */
export interface ShareTarget {
  kind: "document" | "asset"
  key: string
}

/** What a link opens and how. Only the resource's name is shown, never the owner's key. */
export interface SharedResource {
  name: string
  view: "canvas" | "content"
}

const targetQuery = ({ kind, key }: ShareTarget) => toQuery({ kind, key })
const sharedPath = (secret: string) => `/shared/${encodeURIComponent(secret)}`

/** What the target's live link allows, null without one. */
export async function readShareAccess(target: ShareTarget): Promise<ShareAccess | null> {
  return (await request<{ access: ShareAccess | null }>(`/shares${targetQuery(target)}`)).access
}

/** Issue the target's link secret, replacing any previous link. The backend never shows it again. */
export async function createShare(target: ShareTarget): Promise<{ secret: string; access: ShareAccess }> {
  return post<{ secret: string; access: ShareAccess }>("/shares", target)
}

export async function revokeShare(target: ShareTarget): Promise<void> {
  await del(`/shares${targetQuery(target)}`)
}

export async function readSharedResource(secret: string): Promise<SharedResource> {
  return request<SharedResource>(sharedPath(secret))
}

export function sharedContentUrl(secret: string): string {
  return `${getApiBase()}${sharedPath(secret)}/content`
}

export function sharedCanvasEndpoint(secret: string): CanvasEndpoint {
  return `${sharedPath(secret)}/canvas` as CanvasEndpoint
}

/** The page opening a link. The secret rides in the fragment, which browsers never send to the page's host. */
export function shareLinkUrl(secret: string): string {
  return `${window.location.origin}/share#${secret}`
}
