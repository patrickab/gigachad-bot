import { act, renderHook, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useChat } from "@/features/chat/useChat"
import type { ModelsResponse } from "@/lib/domain/types"

const catalog = (tabOrder: string[]) => ({ ollama: [], providers: [], defaults: {}, tab_order: tabOrder }) as unknown as ModelsResponse

const api = vi.hoisted(() => ({
  fetchModels: vi.fn<() => Promise<ModelsResponse>>(),
  fetchPrompts: vi.fn(async () => ({ coder: "You write code." })),
  saveModelTabOrder: vi.fn<(order: string[]) => Promise<ModelsResponse>>(),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({ ...await importOriginal<typeof import("@/lib/backend/api")>(), ...api }))

describe("useChat", () => {
  it("reads models and prompts once for every tab and shows one tab's model save in the others", async () => {
    api.fetchModels.mockResolvedValue(catalog(["Ollama"]))
    api.saveModelTabOrder.mockImplementation(async (order) => catalog(order))
    const first = renderHook(() => useChat())
    const second = renderHook(() => useChat())

    await waitFor(() => expect(second.result.current.models?.tab_order).toEqual(["Ollama"]))
    await waitFor(() => expect(second.result.current.prompts).toEqual({ coder: "You write code." }))
    expect(api.fetchModels).toHaveBeenCalledOnce()
    expect(api.fetchPrompts).toHaveBeenCalledOnce()

    await act(() => first.result.current.saveModelTabOrder(["OpenAI", "Ollama"]))
    expect(second.result.current.models?.tab_order).toEqual(["OpenAI", "Ollama"])
  })
})
