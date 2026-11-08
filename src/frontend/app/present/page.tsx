"use client"

import dynamic from "next/dynamic"
import { useEffect, useSyncExternalStore } from "react"
import { parsePresentQuery } from "@/features/latex/present/presentParams"

// Client only: react-pdf and pdf.js cannot render on the server
const PresentationWindow = dynamic(
  () => import("@/features/latex/present/PresentationWindow").then((m) => ({ default: m.PresentationWindow })),
  { ssr: false },
)

const subscribeNever = () => () => {}

/** A presentation window the LaTeX workbench's Present button opens: `/present?pdf=<url>&role=audience|presenter&channel=<id>`. */
export default function PresentPage() {
  const search = useSyncExternalStore(subscribeNever, () => window.location.search, () => null)
  const params = search === null ? null : parsePresentQuery(search)
  const role = params?.role

  // Tells the two windows apart in the window switcher
  useEffect(() => {
    if (role) document.title = role === "presenter" ? "Presenter" : "Audience"
  }, [role])

  if (search === null) return null
  if (!params) return <main className="flex h-dvh items-center justify-center text-sm text-ink-faint">No presentation</main>
  return <PresentationWindow {...params} />
}
