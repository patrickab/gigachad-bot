import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook } from "@testing-library/react"
import { stageChip, useComposerChips } from "../composerChips"
import { forgetComposerDraft, newChatDraftKey, useComposerDraft } from "../useComposerDraft"

describe("composer drafts", () => {
  let stored: Map<string, string>
  beforeEach(() => {
    stored = new Map()
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => { stored.set(key, value) },
      removeItem: (key: string) => { stored.delete(key) },
    })
  })
  afterEach(() => { cleanup(); vi.unstubAllGlobals() })

  it("keeps a new chat's draft across a remount, by its tab", () => {
    const key = newChatDraftKey("tab-1")
    const first = renderHook(() => useComposerDraft(key))
    act(() => first.result.current[1]("half a prompt"))
    first.unmount()
    expect(renderHook(() => useComposerDraft(key)).result.current[0]).toBe("half a prompt")
    expect(renderHook(() => useComposerDraft(newChatDraftKey("tab-2"))).result.current[0]).toBe("")
  })

  it("forgets the draft and chips of a closed tab", () => {
    const key = newChatDraftKey("tab-1")
    const draft = renderHook(() => useComposerDraft(key))
    const chips = renderHook(() => useComposerChips(key))
    act(() => {
      draft.result.current[1]("half a prompt")
      stageChip(key, { label: "file", text: "contents" })
    })
    expect(stored.size).toBe(2)
    act(() => forgetComposerDraft(key))
    expect(stored.size).toBe(0)
    expect(chips.result.current.chips).toEqual([])
  })
})
