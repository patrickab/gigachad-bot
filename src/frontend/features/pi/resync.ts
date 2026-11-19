/**
 * Resync of one session on the shared pi-host socket (M2, M22).
 *
 * - Tracks the session's `streamId` and the cursor (newest confirmed entry id) from `attached` and `resync` frames
 * - `request` asks for `{type:"resync", sessionId, cursor}`; a guard keeps one request in flight plus one queued follow-up
 * - A frame with a new `streamId` reports `streamChanged` and resyncs from cursor null; responses to older requests or another stream are dropped
 * - A host without resync answers with an `error` frame: from then on every request is a full reattach (`detach` + `attach`)
 * - No ack timeouts (M18): a request in flight is only given up when the socket closes
 */
import type { HostFrame } from "./piSocket"
import type { SessionEntries } from "./transcript"

/*
 * Guard ported from omp-web `lib/reconcile-guard.ts` (https://github.com/ompweb-dev/ompweb, MIT, Copyright (c) 2026 agegr),
 * without its stall timeout: a stalled request is released by the socket closing instead (M18).
 */

/** One resync in flight; triggers during the flight collapse into one follow-up. */
export interface ResyncGuard {
  /** An ownership token when acquired, null when a request is in flight (the trigger is then queued as the follow-up). */
  tryAcquire(): number | null
  /** Whether `token` is the request in flight: false for requests from before a `reset`. */
  owns(token: number): boolean
  /** The request of `token` answered. True when triggers arrived meanwhile and the caller should issue the follow-up. A stale token releases nothing. */
  release(token: number): boolean
  /** Forget the request in flight and the queued follow-up, invalidating every outstanding token. */
  reset(): void
}

export function createResyncGuard(): ResyncGuard {
  let inFlight = false
  let ownerToken = 0
  // Bumped by reset() too, so a pre-reset owner never matches a later token
  let generation = 0
  let pendingWhileInFlight = false

  return {
    tryAcquire() {
      if (inFlight) {
        pendingWhileInFlight = true
        return null
      }
      inFlight = true
      ownerToken = ++generation
      pendingWhileInFlight = false
      return ownerToken
    },
    owns(token) {
      return inFlight && token === ownerToken
    },
    release(token) {
      if (!inFlight || token !== ownerToken) return false
      inFlight = false
      const followUp = pendingWhileInFlight
      pendingWhileInFlight = false
      return followUp
    },
    reset() {
      inFlight = false
      ownerToken = 0
      pendingWhileInFlight = false
      generation += 1
    },
  }
}

interface SentRequest {
  token: number
  kind: "resync" | "attach"
}

/** What the session hook does with a host frame. */
export interface FrameVerdict {
  /** A new `streamId`: drop the live state before applying the frame. */
  streamChanged: boolean
  /** False for a stale `resync` answer and the `error` that switched to reattach: skip the frame. */
  apply: boolean
}

/** Resync state of one session; `send` writes a frame to the shared socket, false while it is down. */
export class SessionResync {
  private readonly guard = createResyncGuard()
  /** Requests sent on this connection, oldest first: the host answers in order. */
  private sent: SentRequest[] = []
  private streamId: string | null = null
  private cursor: string | null = null
  /** The session is attached on this connection (a snapshot arrived, or a reconnect resync went out). */
  private attached = false
  private resyncSupported = true

  private readonly sessionId: string
  private readonly send: (frame: HostFrame) => boolean

  constructor(sessionId: string, send: (frame: HostFrame) => boolean) {
    this.sessionId = sessionId
    this.send = send
  }

  /**
   * Catch up with the host: now, or as the follow-up of the request in flight. No-op until attached.
   * Without host resync a live `message_end` asks for nothing: its message stays live until the next reattach (`agent_settled`) brings it.
   */
  request(trigger: "message_end" | "catch_up" = "catch_up"): void {
    if (!this.attached) return
    if (this.resyncSupported) this.issue("resync")
    else if (trigger === "catch_up") this.issue("attach")
  }

  /** The socket reopened: resync from the cursor when a stream is known, else attach. */
  reconnected(): void {
    if (this.streamId !== null && this.resyncSupported) {
      this.attached = true
      this.issue("resync")
    } else this.issue("attach")
  }

  /** The socket closed: requests in flight will never answer. */
  disconnected(): void {
    this.guard.reset()
    this.sent = []
    this.attached = false
  }

  /** Track one host frame of this session before the hook applies it. */
  receive(frame: HostFrame): FrameVerdict {
    const streamChanged = this.observeStream(frame)
    switch (frame.type) {
      case "attached": {
        this.attached = true
        this.cursor = (frame.entries as SessionEntries).leafId
        if (this.sent[0]?.kind === "attach") this.settle(this.sent.shift()!)
        return { streamChanged, apply: true }
      }
      case "resync": {
        const request = this.sent.shift()
        if (!request || request.kind !== "resync" || !this.guard.owns(request.token)) return { streamChanged, apply: false }
        if (frame.streamId !== this.streamId) {
          // Answered, but stamped for no stream this client follows: free the guard, keep the transcript
          this.guard.release(request.token)
          return { streamChanged, apply: false }
        }
        const entries = frame.entries as SessionEntries["entries"]
        this.cursor = (frame.leafId as string | null) ?? entries[entries.length - 1]?.id ?? this.cursor
        this.settle(request)
        return { streamChanged, apply: true }
      }
      case "error": {
        if (this.sent[0]?.kind !== "resync" || !String(frame.message).includes("resync")) return { streamChanged, apply: true }
        const request = this.sent.shift()!
        this.resyncSupported = false
        this.guard.release(request.token)
        this.issue("attach")
        return { streamChanged, apply: false }
      }
      default:
        return { streamChanged, apply: true }
    }
  }

  /** Adopt the frame's `streamId`. A change after the first one drops every request in flight and resyncs from scratch; an `attached` snapshot is itself from scratch. */
  private observeStream(frame: HostFrame): boolean {
    const streamId = frame.streamId
    if (typeof streamId !== "string" || streamId === this.streamId) return false
    const known = this.streamId !== null
    this.streamId = streamId
    if (!known || frame.type === "attached") return false
    this.guard.reset()
    this.cursor = null
    this.request()
    return true
  }

  private settle(request: SentRequest): void {
    if (this.guard.release(request.token)) this.request()
  }

  private issue(kind: SentRequest["kind"]): void {
    const token = this.guard.tryAcquire()
    if (token === null) return
    const { sessionId } = this
    const sent = kind === "resync"
      ? this.send({ type: "resync", sessionId, cursor: this.cursor })
      // A reattach on an attached connection must detach first, the host attaches a connection once
      : (!this.attached || this.send({ type: "detach", sessionId })) && this.send({ type: "attach", sessionId })
    if (sent) this.sent.push({ token, kind })
    else this.guard.release(token)
  }
}
