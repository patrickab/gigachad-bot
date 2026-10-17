import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

type MockImpl = (...args: unknown[]) => unknown

// Use vi.hoisted so the mutable impl container is initialised before the
// hoisted vi.mock call. The mock factory closes over this container; tests
// mutate it via `apiMock.__setImpl`.
const { impls } = vi.hoisted(() => ({
  impls: {
    listRootFiles: async () => ({ enabled: false, files: [] }),
    attachRootFile: async () => ({ name: "att", mime: "text/plain", url: "u", active: true }),
  } as Record<string, MockImpl>,
}))

vi.mock("@/lib/backend/api", () => ({
  __setImpl: (k: string, fn: (...a: never[]) => unknown) => { impls[k] = fn as MockImpl },
  __resetImpls: () => {
    impls.listRootFiles = async () => ({ enabled: false, files: [] })
    impls.attachRootFile = async () => ({ name: "att", mime: "text/plain", url: "u", active: true })
  },
  listRootFiles: (...a: unknown[]) => impls.listRootFiles(...a),
  attachRootFile: (...a: unknown[]) => impls.attachRootFile(...a),
}))

import * as apiMock from "@/lib/backend/api"

/** Controls the mocked api module exposes beyond the real exports. */
const mockControls = apiMock as unknown as {
  __setImpl: (name: string, impl: (...args: never[]) => unknown) => void
  __resetImpls: () => void
}
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"

import { useFilePicker } from "@/features/chat/useFilePicker"
import { renderHook, act } from "@testing-library/react"

function makeChatInputRef(handle: Partial<ChatInputHandle> = {}) {
  return { current: { addAttachment: vi.fn(), ...handle } as unknown as ChatInputHandle }
}

beforeEach(() => mockControls.__resetImpls())
afterEach(() => vi.useRealTimers())

describe("useFilePicker", () => {
  it("opens standalone PDF view when chat is empty", async () => {
    const { result } = renderHook(() =>
      useFilePicker({
        isActive: true,
        hasMessages: false,
        chatInputRef: makeChatInputRef(),
        setExtracting: vi.fn(),
      }),
    )

    await act(async () => {
      await result.current.handleFileSelect("vault/report.pdf")
    })

    expect(result.current.filePdfPath).toBe("vault/report.pdf")
    expect(result.current.fileEditPath).toBeNull()
    expect(result.current.filePickerOpen).toBe(false)
  })

  it("opens text editor standalone when chat is empty", async () => {
    const { result } = renderHook(() =>
      useFilePicker({
        isActive: true,
        hasMessages: false,
        chatInputRef: makeChatInputRef(),
        setExtracting: vi.fn(),
      }),
    )

    await act(async () => {
      await result.current.handleFileSelect("vault/notes.md")
    })

    expect(result.current.fileEditPath).toBe("vault/notes.md")
    expect(result.current.filePdfPath).toBeNull()
  })

  it("stages attachment on chat input when conversation is active", async () => {
    const addAttachment = vi.fn().mockResolvedValue(undefined)
    const chatInputRef = makeChatInputRef({ addAttachment })
    const attachSpy = vi.fn(async () => ({
      name: "v.txt",
      mime: "text/plain",
      url: "u",
      active: true,
      vaultPath: "vault/v.txt",
    }))
    mockControls.__setImpl("attachRootFile", attachSpy)

    const setExtracting = vi.fn()
    const { result } = renderHook(() =>
      useFilePicker({
        isActive: true,
        hasMessages: true,
        chatInputRef,
        setExtracting,
      }),
    )

    await act(async () => {
      await result.current.handleFileSelect("vault/v.txt")
    })

    expect(attachSpy).toHaveBeenCalledWith("vault/v.txt")
    expect(addAttachment).toHaveBeenCalledOnce()
    expect(result.current.filePdfPath).toBeNull()
    expect(result.current.fileEditPath).toBeNull()
    expect(setExtracting).toHaveBeenCalledTimes(2)
  })

  it("Escape closes the standalone PDF viewer when active", async () => {
    const { result } = renderHook(() =>
      useFilePicker({
        isActive: true,
        hasMessages: false,
        chatInputRef: makeChatInputRef(),
        setExtracting: vi.fn(),
      }),
    )

    await act(async () => {
      await result.current.handleFileSelect("v/x.pdf")
    })
    expect(result.current.filePdfPath).toBe("v/x.pdf")

    await act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    })
    expect(result.current.filePdfPath).toBeNull()
  })
})
