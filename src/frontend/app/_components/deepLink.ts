import type { WorkspaceItem } from "@/contexts/TabContext"

/** What `?open=` asks the shell to show at startup. */
export type DeepLinkTarget =
  | { kind: "chat"; historyFile: string }
  | { kind: "item"; item: WorkspaceItem }

const OPEN_PARAM = "open"

const ITEM_KINDS = ["canvas", "document", "architecture"] as const

function decodeBase64Url(value: string): string {
  // URLSearchParams turns a literal "+" of plain base64 into a space, so accept both alphabets.
  const base64 = value.replace(/ /g, "+").replace(/-/g, "+").replace(/_/g, "/")
  const bytes = Uint8Array.from(atob(base64.padEnd(Math.ceil(base64.length / 4) * 4, "=")), (char) => char.charCodeAt(0))
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes)
}

function targetOf(payload: unknown): DeepLinkTarget | null {
  if (typeof payload !== "object" || payload === null) return null
  const { kind, history_file: historyFile, path, project_slug: projectSlug } = payload as Record<string, unknown>
  if (kind === "chat") return typeof historyFile === "string" && historyFile !== "" ? { kind: "chat", historyFile } : null
  const itemKind = ITEM_KINDS.find((candidate) => candidate === kind)
  if (!itemKind || typeof path !== "string" || path === "") return null
  if (projectSlug != null && typeof projectSlug !== "string") return null
  return { kind: "item", item: { kind: itemKind, path, projectSlug: projectSlug ?? null } }
}

/** Parse the `open` query parameter (base64url JSON); null when absent or malformed. */
export function parseOpenParam(search: string): DeepLinkTarget | null {
  const encoded = new URLSearchParams(search).get(OPEN_PARAM)
  if (!encoded) return null
  try {
    return targetOf(JSON.parse(decodeBase64Url(encoded)))
  } catch {
    return null
  }
}

/** `url`'s path, query and hash without the `open` parameter. */
export function withoutOpenParam(url: URL): string {
  const params = new URLSearchParams(url.search)
  params.delete(OPEN_PARAM)
  const query = params.toString()
  return `${url.pathname}${query ? `?${query}` : ""}${url.hash}`
}
