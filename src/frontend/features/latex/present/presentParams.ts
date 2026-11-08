export type PresentRole = "audience" | "presenter"

/** One presentation window: the PDF it shows, its role and the BroadcastChannel its slide position syncs on. */
export interface PresentParams {
  pdfUrl: string
  role: PresentRole
  channel: string
}

/** The `/present` route of one presentation window. */
export function presentHref({ pdfUrl, role, channel }: PresentParams): string {
  return `/present?${new URLSearchParams({ pdf: pdfUrl, role, channel })}`
}

/** Open one presentation window as a popup, so it can be moved to another screen. Null when the browser blocks it. */
export function openPresentWindow(params: PresentParams): Window | null {
  return window.open(presentHref(params), "_blank", "popup")
}

/** The presentation window a `/present` query string describes, null when it is incomplete. */
export function parsePresentQuery(search: string): PresentParams | null {
  const query = new URLSearchParams(search)
  const pdfUrl = query.get("pdf")
  const role = query.get("role")
  const channel = query.get("channel")
  if (!pdfUrl || !channel || (role !== "audience" && role !== "presenter")) return null
  return { pdfUrl, role, channel }
}
