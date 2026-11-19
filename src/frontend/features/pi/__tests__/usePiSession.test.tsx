import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook } from "@testing-library/react"
import type { HostFrame, SessionSubscriber } from "../piSocket"
import { OPEN_FILE_NOTICE_PREFIX, usePiSession } from "../usePiSession"

const socket = vi.hoisted(() => ({ subscriber: null as SessionSubscriber | null }))
vi.mock("../piSocket", () => ({
  subscribePiSession: (_sessionId: string, subscriber: SessionSubscriber) => {
    socket.subscriber = subscriber
    return () => {}
  },
  sendPiCommand: () => false,
  sendPiFrame: () => true,
}))

function notifyFrame(message: string): HostFrame {
  return { type: "pi", sessionId: "s1", record: { type: "extension_ui_request", id: "u1", method: "notify", message, notifyType: "info" } }
}

describe("usePiSession notify", () => {
  afterEach(() => {
    cleanup()
    socket.subscriber = null
  })

  it("opens an open-file notice's path instead of showing it", () => {
    const onOpenFile = vi.fn()
    const { result } = renderHook(() => usePiSession("s1", onOpenFile))
    act(() => socket.subscriber!.onFrame(notifyFrame(`${OPEN_FILE_NOTICE_PREFIX}/repo/src/app.ts`)))
    expect(onOpenFile).toHaveBeenCalledWith("/repo/src/app.ts")
    expect(result.current.notices).toEqual([])
  })

  it("shows any other notice", () => {
    const onOpenFile = vi.fn()
    const { result } = renderHook(() => usePiSession("s1", onOpenFile))
    act(() => socket.subscriber!.onFrame(notifyFrame("Saved")))
    expect(onOpenFile).not.toHaveBeenCalled()
    expect(result.current.notices).toMatchObject([{ text: "Saved", tone: "info" }])
  })
})
