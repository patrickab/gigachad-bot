/** The memory window follows commits and edits made elsewhere, without overwriting what is typed here. */
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { memoryScopeKey } from "@/features/memory/memoryUtils"
import { editMemory } from "@/features/memory/scopeEdits"
import { useMemoryLibrary } from "@/features/memory/useMemoryLibrary"
import type { StoredMemory } from "@/lib/domain/memory"

type Change = { resource_kind: string; resource_key: string }
const memory = (id: string, text: string) => ({ id, text, topic: "profile", scope: "global" as const, created_at: "t", updated_at: "t" })

const api = vi.hoisted(() => ({
  getTopics: vi.fn(),
  getMemories: vi.fn(),
  saveTopics: vi.fn(async () => undefined),
  saveMemories: vi.fn(async () => undefined),
}))
vi.mock("@/lib/backend/api", () => api)
const feed = vi.hoisted(() => ({ emit: null as null | ((event: Change) => void) }))
vi.mock("@/lib/backend/syncStream", () => ({
  subscribeToChanges: (listener: (event: Change) => void) => { feed.emit = listener; return () => { feed.emit = null } },
}))
vi.mock("@/contexts/ProjectContext", () => ({ useProject: () => ({ projects: [] }) }))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => ({ schedule: vi.fn() }) }))

const GLOBAL_MEMORIES = "memory/global-profile.json"

async function mountLibrary() {
  const hook = renderHook(() => useMemoryLibrary())
  await waitFor(() => expect(hook.result.current.scopes).not.toBeNull())
  return hook
}

async function emitGlobalChange() {
  act(() => feed.emit?.({ resource_kind: "document", resource_key: GLOBAL_MEMORIES }))
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
}

beforeEach(() => {
  vi.clearAllMocks()
  api.getTopics.mockResolvedValue({ topics: [{ name: "profile", summary: "" }] })
  api.getMemories.mockResolvedValue({ memories: [memory("a", "uses Arch")] })
})
afterEach(() => { vi.useRealTimers() })

it("maps memory documents to their scope and ignores review buffers", () => {
  expect(memoryScopeKey("memory/global-topics.json")).toBe("global")
  expect(memoryScopeKey("memory/project/my-proj/memory/memory.json")).toBe("project:my-proj")
  expect(memoryScopeKey("memory/pending/abc.json")).toBeNull()
  expect(memoryScopeKey("memory/watermarks.json")).toBeNull()
})

it("shows a commit made elsewhere", async () => {
  const { result } = await mountLibrary()
  api.getMemories.mockResolvedValue({ memories: [memory("a", "uses Arch"), memory("b", "prefers Neovim")] })

  vi.useFakeTimers()
  await emitGlobalChange()

  expect(result.current.scopeOf("global")?.memories.map((m) => m.text)).toEqual(["uses Arch", "prefers Neovim"])
})

it("keeps a local edit made while the reload was fetching", async () => {
  const { result } = await mountLibrary()
  const fetch = Promise.withResolvers<{ memories: StoredMemory[] }>()
  api.getMemories.mockReturnValue(fetch.promise)

  vi.useFakeTimers()
  await emitGlobalChange()
  act(() => result.current.update("global", editMemory("a", "uses Arch btw")))
  await act(async () => { fetch.resolve({ memories: [memory("a", "stale server text")] }) })

  expect(result.current.scopeOf("global")?.memories[0].text).toBe("uses Arch btw")
  expect(api.saveMemories).toHaveBeenLastCalledWith("global", [expect.objectContaining({ text: "uses Arch btw" })], null)
})
