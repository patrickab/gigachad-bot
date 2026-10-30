import { useCallback } from "react"
import type { LearnSession } from "@/lib/domain/types"
import type { LearnPanel } from "@gigachad/chat-ui/learn-protocol"
import { useLearn, useLearnFrameState } from "./LearnProvider"

/** LearnHarness's per-session popups: whether a session has a plan, and opening a popup (Plan, Coverage) on its session in the frame. */
export function useLearnPanels() {
  const { showInFrame } = useLearn()
  const plans = useLearnFrameState("session-plan")
  const hasPlan = (sessionId: string) => plans[sessionId]?.available === true
  const openPanel = useCallback((session: Pick<LearnSession, "id" | "project_slug">, panel: LearnPanel) => {
    showInFrame({ type: "open-panel", panel, sessionId: session.id }, session)
  }, [showInFrame])
  return { hasPlan, openPanel }
}
