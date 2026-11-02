import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, renderHook } from "@testing-library/react"
import { announceFileClosed } from "@/lib/backend/fileEvents"
import { useKernelShutdown } from "../useKernelShutdown"

const api = vi.hoisted(() => ({ shutdownKernel: vi.fn<(path: string) => Promise<void>>() }))
vi.mock("@/lib/backend/api", () => api)
vi.mock("@/lib/backend/syncStream", () => ({ subscribeToChanges: () => () => {} }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe("useKernelShutdown", () => {
  it("shuts the kernel of a closed notebook down", () => {
    api.shutdownKernel.mockResolvedValue()
    renderHook(() => useKernelShutdown())
    announceFileClosed("/work/a.ipynb")
    expect(api.shutdownKernel).toHaveBeenCalledWith("/work/a.ipynb")
  })

  it("leaves other closed files alone", () => {
    renderHook(() => useKernelShutdown())
    announceFileClosed("/work/notes.md")
    expect(api.shutdownKernel).not.toHaveBeenCalled()
  })

  it("survives a failed shutdown and stops listening when unmounted", async () => {
    api.shutdownKernel.mockRejectedValue(new Error("offline"))
    const { unmount } = renderHook(() => useKernelShutdown())
    announceFileClosed("/work/a.ipynb")
    await Promise.resolve()
    unmount()
    announceFileClosed("/work/b.ipynb")
    expect(api.shutdownKernel).toHaveBeenCalledTimes(1)
  })
})
