"use client"

import { useEffect, useState, type ReactNode } from "react"
import { setApiBase } from "@/lib/backend/config"

interface BackendInfo {
  baseUrl: string
}

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window
}

/** Holds `children` until the Tauri sidecar reports its backend URL. Outside Tauri it renders them at once. */
export function DesktopBackendProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<"starting" | "ready" | "error">(isTauri() ? "starting" : "ready")
  const [error, setError] = useState("")

  useEffect(() => {
    if (!isTauri()) return

    // Dynamic import keeps the Tauri chunk out of the web bundle entirely.
    import("@tauri-apps/api/core")
      .then(({ invoke }) => invoke<BackendInfo>("backend_info"))
      .then(({ baseUrl }) => {
        setApiBase(`${baseUrl}/api`)
        setState("ready")
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : String(reason))
        setState("error")
      })
  }, [])

  if (state === "ready") return children

  return (
    <main className="flex h-dvh items-center justify-center bg-paper px-6 text-ink">
      <div className="max-w-md space-y-3 text-center">
        <p className="text-lg font-medium">{state === "starting" ? "Starting GigaChad Bot…" : "GigaChad Bot could not start"}</p>
        {state === "error" && <p className="text-sm text-ink/70">{error}</p>}
      </div>
    </main>
  )
}
