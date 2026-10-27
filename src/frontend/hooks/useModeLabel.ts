"use client"

import { useEffect, useEffectEvent } from "react"

/** Announce `label` to the host through `onModeLabel` whenever `label` changes or `enabled` turns true. */
export function useModeLabel(label: string, enabled: boolean, onModeLabel?: (label: string) => void): void {
  const announce = useEffectEvent(() => { if (enabled) onModeLabel?.(label) })
  useEffect(() => { announce() }, [label, enabled])
}
