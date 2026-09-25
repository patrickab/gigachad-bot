"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { ApiError, loadCanvasSnapshot, submitCanvasMutations, type CanvasAcceptedBatch, type CanvasEntity, type CanvasMutation } from "@/lib/api"
import { subscribeToCanvasMutationBatches } from "@/lib/canvasStream"
import type { CanvasDocument } from "@/components/CanvasEditor"

export type CanvasCollaborationStatus = "loading" | "synced" | "saving" | "offline" | "error"

export interface CollaborativeCanvas {
  document: CanvasDocument | null
  replace: (nextDocument: CanvasDocument) => void
  status: CanvasCollaborationStatus
  retry: () => void
}

const SEND_DEBOUNCE_MS = 75
const RETRY_INITIAL_MS = 250
const RETRY_MAX_MS = 5_000
const RETRY_MAX_ATTEMPTS = 5

type Entity = { id: string }
type CollectionName = "strokes" | "frames" | "attachments" | "texts"
const COLLECTIONS: [CanvasEntity, CollectionName][] = [["stroke", "strokes"], ["frame", "frames"], ["attachment", "attachments"], ["text", "texts"]]
const COLLECTION: Record<CanvasEntity, CollectionName> = Object.fromEntries(COLLECTIONS) as Record<CanvasEntity, CollectionName>

/** All mutable state of one loaded canvas. A new canvas key gets a new session, so
 * late callbacks from the previous one are recognised and dropped. */
type Session = {
  canvasKey: string
  server: CanvasDocument | null
  revision: number
  viewport: CanvasDocument["viewport"]
  pending: CanvasMutation[]
  /** Sent mutations; kept as an overlay until the stream delivers `ackRevision`. */
  inFlight: { mutations: CanvasMutation[]; ackRevision?: number } | null
  attempts: number
  timer?: NodeJS.Timeout | number
  closeStream?: () => void
}

function deriveMutations(previous: CanvasDocument, next: CanvasDocument): CanvasMutation[] {
  const mutations: CanvasMutation[] = []
  for (const [entity, collection] of COLLECTIONS) {
    if (previous[collection] === next[collection]) continue
    const before = new Map((previous[collection] as Entity[]).map((value) => [value.id, value]))
    for (const value of next[collection] as Entity[]) {
      const prior = before.get(value.id)
      before.delete(value.id)
      if (prior !== value && JSON.stringify(prior) !== JSON.stringify(value)) {
        mutations.push({ mutationId: crypto.randomUUID(), entity, kind: "upsert", id: value.id, value: value as unknown as Record<string, unknown> })
      }
    }
    for (const id of before.keys()) mutations.push({ mutationId: crypto.randomUUID(), entity, kind: "delete", id })
  }
  return mutations
}

/** Keep only the latest pending mutation per entity. */
function compact(mutations: CanvasMutation[]): CanvasMutation[] {
  const latest = new Map(mutations.map((change) => [`${change.entity}:${change.id}`, change]))
  return mutations.filter((change) => latest.get(`${change.entity}:${change.id}`) === change)
}

function applyMutations(document: CanvasDocument, mutations: CanvasMutation[]): CanvasDocument {
  const next = { ...document }
  for (const change of mutations) {
    const collection = COLLECTION[change.entity]
    const values = [...(next[collection] as Entity[])]
    const index = values.findIndex((value) => value.id === change.id)
    if (change.kind === "delete") {
      if (index !== -1) values.splice(index, 1)
    } else if (index === -1) {
      values.push({ ...change.value, id: change.id })
    } else {
      values[index] = { ...change.value, id: change.id }
    }
    ;(next as Record<CollectionName, unknown>)[collection] = values
  }
  return next
}

function failureStatus(): CanvasCollaborationStatus {
  return typeof navigator !== "undefined" && navigator.onLine === false ? "offline" : "error"
}

/**
 * Server-sequenced collaboration for one `.canvas` document. POSTs only acknowledge
 * the revision holding a batch; the gapless stream is the single source of content.
 */
export function useCollaborativeCanvas(canvasKey: string): CollaborativeCanvas {
  const [document, setDocument] = useState<CanvasDocument | null>(null)
  const [status, setStatus] = useState<CanvasCollaborationStatus>("loading")
  const documentRef = useRef<CanvasDocument | null>(null)
  const sessionRef = useRef<Session | null>(null)

  const publish = useCallback((session: Session) => {
    if (sessionRef.current !== session || !session.server) return
    const base = applyMutations(session.server, session.inFlight?.mutations ?? [])
    documentRef.current = { ...applyMutations(base, session.pending), viewport: session.viewport }
    setDocument(documentRef.current)
    setStatus(session.inFlight || session.pending.length > 0 ? "saving" : "synced")
  }, [])

  const run = useCallback((session: Session, action: () => void, delay: number) => {
    clearTimeout(session.timer)
    session.timer = setTimeout(() => {
      if (sessionRef.current === session) action()
    }, delay)
  }, [])

  const fail = useCallback((session: Session, error: unknown, action: () => void) => {
    setStatus(failureStatus())
    const transient = !(error instanceof ApiError) || error.status === 408 || error.status === 429 || error.status >= 500
    if (!transient || session.attempts >= RETRY_MAX_ATTEMPTS) return
    run(session, action, Math.min(RETRY_INITIAL_MS * 2 ** session.attempts, RETRY_MAX_MS))
    session.attempts += 1
  }, [run])

  const send = useCallback((session: Session) => {
    if (!session.server || session.inFlight || session.pending.length === 0) return
    const inFlight = { mutations: session.pending }
    session.inFlight = inFlight
    session.pending = []
    submitCanvasMutations(session.canvasKey, inFlight.mutations).then(({ revision }) => {
      if (session.inFlight !== inFlight) return
      session.attempts = 0
      if (session.revision >= revision) {
        session.inFlight = null
        send(session)
        publish(session)
      } else {
        session.inFlight = { ...inFlight, ackRevision: revision }
      }
    }).catch((error: unknown) => {
      if (session.inFlight !== inFlight) return
      // Mutation IDs are idempotent, so resending ones the server already holds is safe.
      session.inFlight = null
      session.pending = compact([...inFlight.mutations, ...session.pending])
      publish(session)
      fail(session, error, () => send(session))
    })
  }, [fail, publish])

  const receiveBatch = useCallback((session: Session, batch: CanvasAcceptedBatch) => {
    if (!session.server || batch.revision <= session.revision) return
    session.server = applyMutations(session.server, batch.mutations)
    session.revision = batch.revision
    const ack = session.inFlight?.ackRevision
    if (ack !== undefined && ack <= batch.revision) {
      session.inFlight = null
      send(session)
    }
    publish(session)
  }, [publish, send])

  const load = useCallback((session: Session) => {
    loadCanvasSnapshot(session.canvasKey).then(({ revision, document: snapshot }) => {
      if (sessionRef.current !== session) return
      session.attempts = 0
      session.server = snapshot
      session.revision = revision
      session.viewport ??= snapshot.viewport
      session.closeStream = subscribeToCanvasMutationBatches(
        session.canvasKey,
        () => session.revision,
        (batch) => receiveBatch(session, batch),
        () => setStatus(failureStatus()),
        () => publish(session),
      )
      publish(session)
      send(session)
    }).catch((error: unknown) => {
      if (sessionRef.current === session) fail(session, error, () => load(session))
    })
  }, [fail, publish, receiveBatch, send])

  useEffect(() => {
    const session: Session = { canvasKey, server: null, revision: 0, viewport: undefined, pending: [], inFlight: null, attempts: 0 }
    sessionRef.current = session
    documentRef.current = null
    setDocument(null)
    setStatus("loading")
    load(session)
    return () => {
      clearTimeout(session.timer)
      send(session) // flush edits still inside the debounce window
      sessionRef.current = null
      session.closeStream?.()
    }
  }, [canvasKey, load, send])

  const retry = useCallback(() => {
    const session = sessionRef.current
    if (!session) return
    session.attempts = 0
    clearTimeout(session.timer)
    if (!session.server) {
      setStatus("loading")
      load(session)
    } else {
      send(session)
      publish(session)
    }
  }, [load, publish, send])

  useEffect(() => {
    const offline = () => setStatus("offline")
    window.addEventListener("offline", offline)
    window.addEventListener("online", retry)
    return () => {
      window.removeEventListener("offline", offline)
      window.removeEventListener("online", retry)
    }
  }, [retry])

  const replace = useCallback((nextDocument: CanvasDocument) => {
    const session = sessionRef.current
    const current = documentRef.current
    if (!session?.server || !current) return
    session.viewport = nextDocument.viewport
    const changes = deriveMutations(current, nextDocument)
    documentRef.current = nextDocument
    setDocument(nextDocument)
    if (changes.length === 0) return
    session.pending = compact([...session.pending, ...changes])
    setStatus("saving")
    run(session, () => send(session), SEND_DEBOUNCE_MS)
  }, [run, send])

  return { document, replace, status, retry }
}
