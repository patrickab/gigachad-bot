import type { Message } from "@/lib/domain/types"

/** One question and its answer. Messages alternate user, assistant, so a pair starts at an even message index. */
export interface Pair {
  user: Message
  assistant: Message
  /** Message index of `user`. */
  globalIndex: number
  /** Position among the pairs. */
  qaIndex: number
}

/** Pair user/assistant messages, padding a missing assistant reply with an empty one. */
export function pairMessages(messages: Message[]): Pair[] {
  const pairs: Pair[] = []
  for (let i = 0; i < messages.length; i += 2) {
    if (messages[i]?.role !== "user") continue
    pairs.push({ user: messages[i], assistant: messages[i + 1] ?? { role: "assistant", content: "" }, globalIndex: i, qaIndex: pairs.length })
  }
  return pairs
}

/** Stable keys of the pairs of one message list, and what the next list's keys derive from. */
export interface PairKeys {
  keys: string[]
  byMessage: ReadonlyMap<Message, string>
  next: number
}

/**
 * A key per pair that survives the delete of an earlier pair and the replacement of one of its messages (a streamed answer,
 * a toggled attachment): a pair keeps the key of the earlier pair that shares a message with it, else it takes a new one.
 */
export function keyPairs(pairs: readonly Pair[], previous?: PairKeys): PairKeys {
  const byMessage = new Map<Message, string>()
  const used = new Set<string>()
  let next = previous?.next ?? 0
  const keys = pairs.map(({ user, assistant }) => {
    let key = previous?.byMessage.get(user) ?? previous?.byMessage.get(assistant)
    if (key === undefined || used.has(key)) key = `pair-${next++}`
    used.add(key)
    byMessage.set(user, key)
    byMessage.set(assistant, key)
    return key
  })
  return { keys, byMessage, next }
}
