import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

// Use vi.hoisted so the mutable impl container is initialised before the
// hoisted vi.mock call. The mock factory closes over this container; tests
// mutate it via `apiMock.__setImpl`.
const { impls } = vi.hoisted(() => ({
  impls: {
    listFileVaultFiles: async () => ({ enabled: false, files: [] }),
    attachFileVaultFile: async () => ({ name: "att", mime: "text/plain", url: "u", active: true }),
  } as Record<string, (...a: any[]) => any>,
}))

vi.mock("@/lib/api", () => ({
  __setImpl: (k: string, fn: any) => { impls[k] = fn },
  __resetImpls: () => {
    impls.listFileVaultFiles = async () => ({ enabled: false, files: [] })
    impls.attachFileVaultFile = async () => ({ name: "att", mime: "text/plain", url: "u", active: true })
  },
  listFileVaultFiles: (...a: any[]) => impls.listFileVaultFiles(...a),
  attachFileVaultFile: (...a: any[]) => impls.attachFileVaultFile(...a),
}))

import * as apiMock from "@/lib/api"
import type { ChatInputHandle } from "@/components/ChatInput"

import { useVaultPicker } from "@/hooks/useVaultPicker"
import { renderHook, act, waitFor } from "@testing-library/react"

function makeChatInputRef(handle: Partial<ChatInputHandle> = {}) {
  return { current: { addAttachment: vi.fn(), ...handle } as unknown as ChatInputHandle }
}

beforeEach(() => (apiMock as any).__resetImpls())
afterEach(() => vi.useRealTimers())

describe("useVaultPicker", () => {
  it("loads vault files on mount", async () => {
    const listSpy = vi.fn(async () => ({
      enabled: true,
      files: [{ path: "v/a.txt", name: "a.txt" }],
    }))
    ;(apiMock as any).__setImpl("listFileVaultFiles", listSpy)

    const { result } = renderHook(() =>
      useVaultPicker({
        isActive: true,
        hasMessages: false,
        chatInputRef: makeChatInputRef(),
        setExtracting: vi.fn(),
      }),
    )

    await waitFor(() => expect(result.current.vaultEnabled).toBe(true))
    expect(result.current.vaultFiles).toHaveLength(1)
    expect(listSpy).toHaveBeenCalledOnce()
  })

  it("opens standalone PDF view when chat is empty", async () => {
    const { result } = renderHook(() =>
      useVaultPicker({
        isActive: true,
        hasMessages: false,
        chatInputRef: makeChatInputRef(),
        setExtracting: vi.fn(),
      }),
    )

    await act(async () => {
      await result.current.handleVaultSelect("vault/report.pdf")
    })

    expect(result.current.vaultPdfPath).toBe("vault/report.pdf")
    expect(result.current.vaultEditPath).toBeNull()
    expect(result.current.vaultPickerOpen).toBe(false)
  })

  it("opens text editor standalone when chat is empty", async () => {
    const { result } = renderHook(() =>
      useVaultPicker({
        isActive: true,
        hasMessages: false,
        chatInputRef: makeChatInputRef(),
        setExtracting: vi.fn(),
      }),
    )

    await act(async () => {
      await result.current.handleVaultSelect("vault/notes.md")
    })

    expect(result.current.vaultEditPath).toBe("vault/notes.md")
    expect(result.current.vaultPdfPath).toBeNull()
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
    ;(apiMock as any).__setImpl("attachFileVaultFile", attachSpy)

    const setExtracting = vi.fn()
    const { result } = renderHook(() =>
      useVaultPicker({
        isActive: true,
        hasMessages: true,
        chatInputRef,
        setExtracting,
      }),
    )

    await act(async () => {
      await result.current.handleVaultSelect("vault/v.txt")
    })

    expect(attachSpy).toHaveBeenCalledWith("vault/v.txt")
    expect(addAttachment).toHaveBeenCalledOnce()
    expect(result.current.vaultPdfPath).toBeNull()
    expect(result.current.vaultEditPath).toBeNull()
    // extracting counter bumped up then back down (functional updates).
    expect(setExtracting).toHaveBeenCalledTimes(2)
  })

  it("does not attach when the API throws (swallowed, extracting restored)", async () => {
    const addAttachment = vi.fn()
    const chatInputRef = makeChatInputRef({ addAttachment })
    ;(apiMock as any).__setImpl("attachFileVaultFile", async () => {
      throw new Error("network")
    })

    const setExtracting = vi.fn()
    const { result } = renderHook(() =>
      useVaultPicker({
        isActive: true,
        hasMessages: true,
        chatInputRef,
        setExtracting,
      }),
    )

    await act(async () => {
      await result.current.handleVaultSelect("vault/v.txt")
    })

    expect(addAttachment).not.toHaveBeenCalled()
    expect(setExtracting).toHaveBeenCalledTimes(2) // up then down
  })

  it("openVaultPicker opens the picker and refreshes the list", async () => {
    const listSpy = vi.fn(async () => ({ enabled: false, files: [] }))
    ;(apiMock as any).__setImpl("listFileVaultFiles", listSpy)

    const { result } = renderHook(() =>
      useVaultPicker({
        isActive: true,
        hasMessages: false,
        chatInputRef: makeChatInputRef(),
        setExtracting: vi.fn(),
      }),
    )

    await act(() => result.current.openVaultPicker())

    expect(result.current.vaultPickerOpen).toBe(true)
    // Initial mount call + refresh on open.
    expect(listSpy).toHaveBeenCalledTimes(2)
  })

  it("Escape closes the standalone PDF viewer when active", async () => {
    const { result } = renderHook(() =>
      useVaultPicker({
        isActive: true,
        hasMessages: false,
        chatInputRef: makeChatInputRef(),
        setExtracting: vi.fn(),
      }),
    )

    await act(async () => {
      await result.current.handleVaultSelect("v/x.pdf")
    })
    expect(result.current.vaultPdfPath).toBe("v/x.pdf")

    await act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    })
    expect(result.current.vaultPdfPath).toBeNull()
  })
})
