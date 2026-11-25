/** The memory window lists memory_stores, opens a memory and writes fact edits as memory-store ops. */
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { useMemoryLibrary } from "@/features/memory/useMemoryLibrary"
import type { Fact, MemoryListing, MemoryOp } from "@/lib/domain/memory"

const api = vi.hoisted(() => ({
  listMemories: vi.fn(),
  readMemory: vi.fn(),
  applyMemoryOps: vi.fn(),
}))
vi.mock("@/lib/backend/api", () => api)
const undo = vi.hoisted(() => ({
  schedule: vi.fn((_key: string, _label: string, run: () => Promise<void>) => { void run() }),
  pending: vi.fn<(key: string) => boolean>(() => false),
}))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => undo }))

const fact = (id: string, text: string, updated: string): Fact => ({ id, text, created: updated, updated })
const listing = (overrides: Partial<MemoryListing> = {}): MemoryListing => ({ memory_stores: [], directories: [], memories: [], documents: [], ...overrides })

beforeEach(() => {
  vi.clearAllMocks()
  api.listMemories.mockImplementation(async (memoryStore = "") => memoryStore === ""
    ? listing({ memory_stores: ["user"] })
    : listing({ memories: [{ path: "profile", description: "Who the user is", facts: 2, updated: "2026-01-02T00:00:00Z" }] }))
  api.readMemory.mockResolvedValue({
    kind: "memory",
    description: "Who the user is",
    facts: [fact("a1", "uses Arch", "2026-01-01T00:00:00Z"), fact("b2", "prefers Neovim", "2026-01-02T00:00:00Z")],
  })
  api.applyMemoryOps.mockImplementation(async (_store: string, ops: MemoryOp[]) =>
    ops.map((op) => ({ ok: true, op: op.op, path: op.path, id: op.op === "add" || op.op === "update" ? "n3w" : undefined })))
})

async function openProfile() {
  const hook = renderHook(() => useMemoryLibrary())
  await waitFor(() => expect(hook.result.current.sections).not.toBeNull())
  act(() => hook.result.current.openPage("user", "profile"))
  await waitFor(() => expect(hook.result.current.page).not.toBeNull())
  return hook
}

const shownFacts = (page: ReturnType<typeof useMemoryLibrary>["page"]) => page?.kind === "memory" ? page.facts.map((f) => f.text) : null

it("lists each top-level memory_store with its memories", async () => {
  const { result } = renderHook(() => useMemoryLibrary())
  await waitFor(() => expect(result.current.sections).not.toBeNull())

  expect(api.listMemories).toHaveBeenCalledWith()
  expect(api.listMemories).toHaveBeenCalledWith("user")
  expect(result.current.sections?.map((s) => [s.memoryStore, s.listing.memories.map((m) => m.description)])).toEqual([["user", ["Who the user is"]]])
})

it("opens a memory with its facts newest first", async () => {
  const { result } = await openProfile()

  expect(api.readMemory).toHaveBeenCalledWith("user", "profile")
  expect(shownFacts(result.current.page)).toEqual(["prefers Neovim", "uses Arch"])
})

it("writes an edited fact as an update op", async () => {
  const { result } = await openProfile()

  act(() => result.current.update("a1", "uses Arch btw"))

  expect(shownFacts(result.current.page)).toContain("uses Arch btw")
  await waitFor(() => expect(api.applyMemoryOps).toHaveBeenCalledWith("user", [{ op: "update", path: "profile", id: "a1", text: "uses Arch btw" }]))
})

it("writes a deleted fact as a retire op through the undo window", async () => {
  const { result } = await openProfile()

  act(() => result.current.retire("b2"))

  expect(shownFacts(result.current.page)).toEqual(["uses Arch"])
  expect(undo.schedule).toHaveBeenCalledWith("memory-fact:user:profile:b2", "fact", expect.any(Function), expect.any(Function))
  await waitFor(() => expect(api.applyMemoryOps).toHaveBeenCalledWith("user", [{ op: "retire", path: "profile", id: "b2" }]))
})

it("writes a new fact as an add op and shows it first", async () => {
  const { result } = await openProfile()

  act(() => result.current.add("works at night"))

  expect(shownFacts(result.current.page)?.[0]).toBe("works at night")
  await waitFor(() => expect(api.applyMemoryOps).toHaveBeenCalledWith("user", [{ op: "add", path: "profile", text: "works at night" }]))
  await waitFor(() => expect(api.readMemory).toHaveBeenCalledTimes(2))
})

it("reports a failed op", async () => {
  const { result } = await openProfile()
  api.applyMemoryOps.mockResolvedValueOnce([{ ok: false, op: "describe", path: "profile", error: "invalid_op", message: "description is empty" }])

  act(() => result.current.describe("x"))

  await waitFor(() => expect(result.current.error).toBe("Could not save memory: description is empty"))
})

it("a second edit before the read uses the id the first edit stored", async () => {
  const { result } = await openProfile()
  let stored = 0
  api.applyMemoryOps.mockImplementation(async (_store: string, ops: MemoryOp[]) =>
    ops.map((op) => ({ ok: true, op: op.op, path: op.path, id: `v${++stored}` })))

  act(() => result.current.update("a1", "uses Arch btw"))
  act(() => result.current.update("a1", "uses Arch, really"))

  await waitFor(() => expect(api.applyMemoryOps).toHaveBeenCalledTimes(2))
  expect(api.applyMemoryOps).toHaveBeenLastCalledWith("user", [{ op: "update", path: "profile", id: "v1", text: "uses Arch, really" }])
})

it("a read during the undo window keeps the retired fact hidden", async () => {
  const { result } = await openProfile()
  undo.schedule.mockImplementationOnce(() => {})
  undo.pending.mockImplementation((key: string) => key === "memory-fact:user:profile:b2")

  act(() => result.current.retire("b2"))
  act(() => result.current.add("works at night"))

  await waitFor(() => expect(api.readMemory).toHaveBeenCalledTimes(2))
  await waitFor(() => expect(shownFacts(result.current.page)).toEqual(["uses Arch"]))
})
