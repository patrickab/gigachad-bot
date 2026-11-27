"use client"

import { useEffect, useEffectEvent, useMemo } from "react"
import { useStore } from "@/hooks/useStore"
import { liveSession } from "./liveSession"
import { useHeld } from "./outbox"
import type { Transcript } from "./transcript"

/** Hold session `sessionId` attached while mounted (`liveSession.ts`): its state, held messages and commands. */
export function usePiSession(sessionId: string, onOpenFile: (path: string) => void) {
  const live = liveSession(sessionId)
  const { btwThreads, ...state } = useStore(live.state)
  const held = useHeld(sessionId)
  const openFile = useEffectEvent(onOpenFile)

  useEffect(() => {
    const stopOpening = live.onOpenFile((path) => openFile(path))
    const release = live.attach()
    return () => {
      release()
      stopOpening()
    }
  }, [live])

  const btw = useMemo(() => ({
    threads: btwThreads,
    followUp: (id: string, question: string) => void live.send(`/btw-followup ${id} ${question}`, []),
    cancel: (id: string) => void live.send(`/btw-cancel ${id}`, []),
    dismiss: (id: string) => void live.send(`/btw-dismiss ${id}`, []),
  }), [btwThreads, live])

  return {
    ...state, btw, held, retryHeld: live.retryHeld, deleteHeld: live.deleteHeld, send: live.send, abort: live.abort, runBash: live.runBash,
    compact: live.compact, setModel: live.setModel, setThinking: live.setThinking, answerDialog: live.answerDialog,
    inspectSubagent: live.inspectSubagent, notify: live.notify,
  }
}

/** The session's transcript as `select` reads it. */
export function usePiTranscript<T = Transcript>(sessionId: string, select?: (transcript: Transcript) => T): T {
  return useStore(liveSession(sessionId).transcript, select)
}
