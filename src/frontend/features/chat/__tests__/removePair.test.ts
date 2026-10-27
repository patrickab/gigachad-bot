import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useChatStream } from "@/features/chat/useChatStream"
import type { Message } from "@/lib/domain/types"

vi.mock("@/lib/backend/api", () => ({ createChatStream: vi.fn() }))

function turn(n: number): [Message, Message] {
  return [{ role: "user", content: `question ${n}` }, { role: "assistant", content: `answer ${n}` }]
}

/** A chat holding `turns` pairs, and the restore returned by removing pair `removed` (1-based). */
function chatWithRemovedPair(turns: number, removed: number) {
  const pairs = Array.from({ length: turns }, (_, i) => turn(i + 1))
  const { result } = renderHook(() => useChatStream())
  act(() => { result.current.replaceMessages(() => pairs.flat()) })
  let restore = null as (() => void) | null
  act(() => { restore = result.current.removePair(pairs[removed - 1][0]) })
  if (!restore) throw new Error("removePair found no pair")
  return { result, pairs, restore }
}

describe("removePair restore", () => {
  it("lands at the original index when the follower message was replaced during the undo window", () => {
    const { result, pairs, restore } = chatWithRemovedPair(3, 2)
    const [[user1, answer1], [user2, answer2], [user3, answer3]] = pairs
    const editedFollower: Message = { ...user3, content: "question 3, edited" }
    act(() => { result.current.replaceMessages((current) => current.map((m) => m === user3 ? editedFollower : m)) })

    act(() => restore())

    expect(result.current.messages).toEqual([user1, answer1, user2, answer2, editedFollower, answer3])
    expect(result.current.messages[2]).toBe(user2)
    expect(result.current.messages[3]).toBe(answer2)
  })

  it("lands at the end-clamped index when the chat shrank during the undo window", () => {
    const { result, pairs, restore } = chatWithRemovedPair(4, 3)
    const [[user1, answer1], , [user3, answer3]] = pairs
    act(() => { result.current.replaceMessages(() => [user1, answer1]) })

    act(() => restore())

    expect(result.current.messages).toEqual([user1, answer1, user3, answer3])
    expect(result.current.messages[2]).toBe(user3)
  })
})
