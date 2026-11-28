"use client"

import { useEffect, useState } from "react"
import { useTabId } from "@/contexts/TabContext"

/**
 * False for the one commit in which the tab in front changed, else true. A quiet read enabled by it starts over then, so a
 * failed first read is tried again on every tab switch, as when each tab's own sidebar came to the front.
 */
export function useSettledTab(): boolean {
  const tabId = useTabId()
  const [settled, setSettled] = useState(tabId)
  // eslint-disable-next-line react-hooks/set-state-in-effect -- the unsettled commit is the point: it turns a quiet read off and on again
  useEffect(() => setSettled(tabId), [tabId])
  return settled === tabId
}
