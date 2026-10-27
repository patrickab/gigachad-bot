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

/** Message index of the user message that starts the pair at `qaIndex`. */
export function pairStart(qaIndex: number): number {
  return qaIndex * 2
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
