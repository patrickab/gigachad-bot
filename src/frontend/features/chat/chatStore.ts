import { createChatStream, currentConnectionPool, leaveConnectionPool } from "@/lib/backend/api"
import { deactivateSentImages } from "@/features/chat/attachments"
import { createTokenPacer } from "@/lib/backend/tokenPacer"
import { applyChatStreamEvent, decodeChatStreamEvent, settleRunningToolCalls } from "@/lib/backend/streaming"
import { toolPayload } from "@/features/chat/toolCalls"
import { createStore, type Store } from "@/lib/store"
import type { ChatRequest, Message, Usage } from "@/lib/domain/types"
import { errorMessage } from "@/lib/utils"

export interface ChatState { messages: Message[]; isStreaming: boolean; usage: Usage }
export interface ChatStreamCompletion { messages: Message[]; usage: Usage }

export interface ChatStore extends Pick<Store<ChatState>, "get" | "subscribe"> {
  /** Replace the messages by `change(current)` at once: the next `get()`, and a `send` right after, see them. Returns them. */
  update: (change: (current: Message[]) => Message[]) => Message[]
  /** Replace messages and usage in one change, by an empty chat unless given. */
  reset: (messages?: Message[], usage?: Usage) => void
  /**
   * Stream one reply into `placeholder`, the empty answer the caller appended after its question. Resolves to null when
   * `placeholder` is not the last message, or when cancelled, failed or not completed.
   */
  send: (req: ChatRequest, placeholder: Message) => Promise<ChatStreamCompletion | null>
  cancel: () => void
}

/** Build the API history, prepending hidden attachment text to user turns and appending mindmap output to assistant turns. */
function buildHistory(msgs: Message[]): { role: string; content: string }[] {
  return msgs
    .filter(m => m.role === "user" || m.role === "assistant")
    .map(m => {
      let content = m.content
      if (m.role === "user" && m.hiddenContent) {
        content = m.hiddenContent + "\n\n" + content
      }
      const mindmap = m.tool_calls?.map(toolPayload).find((payload) => payload.family === "mindmap")
      if (mindmap?.family === "mindmap" && mindmap.content) content = [content, mindmap.content].filter(Boolean).join("\n\n")
      return { role: m.role, content }
    })
}

/** `messages` without the pair whose question is `user` (`"pair"`), or without it and every later one (`"end"`); null when `user` is not there. */
export function withoutPairs(messages: Message[], user: Message, through: "pair" | "end"): Message[] | null {
  const index = messages.indexOf(user)
  if (index < 0) return null
  // A pair whose answer is missing is the user message alone
  const end = through === "end" ? messages.length : index + (messages[index + 1]?.role === "assistant" ? 2 : 1)
  return [...messages.slice(0, index), ...messages.slice(end)]
}

const NO_USAGE: Usage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }

/** A chat's messages and usage, and the one assistant reply streaming into them at a time. */
export function createChatStore(): ChatStore {
  const store = createStore<ChatState>({ messages: [], isStreaming: false, usage: NO_USAGE })
  let abort: (() => void) | null = null

  const update = (change: (current: Message[]) => Message[]) => {
    const messages = change(store.get().messages)
    if (messages === store.get().messages) return messages
    store.set({ ...store.get(), messages })
    return messages
  }
  const setUsage = (usage: Usage) => store.set({ ...store.get(), usage })
  const setStreaming = (isStreaming: boolean) => store.set({ ...store.get(), isStreaming })

  const send = async (req: ChatRequest, placeholder: Message): Promise<ChatStreamCompletion | null> => {
    // A stream writes its answer only while that answer is still the chat's last message: a reset, another chat loaded
    // or the pair deleted leave the list alone. `shown` is the copy of the accumulator last put into the list.
    let shown = placeholder
    if (store.get().messages.at(-1) !== shown) return null
    const answer: Message = { role: "assistant", content: "" }
    const history = buildHistory(store.get().messages.slice(0, -2))
    setStreaming(true)
    const publish = () => {
      if (store.get().messages.at(-1) !== shown) return
      shown = { ...answer }
      update((current) => [...current.slice(0, -1), shown])
    }
    // Answer text is paced, everything else lands as it arrives.
    const pacer = createTokenPacer((text) => {
      answer.content += text
      publish()
    })
    // Every exit from the loop below settles the tool cards still marked running.
    let unresolvedReason = "Stream ended before the tool returned"
    let completed = false
    let usage = store.get().usage
    let finalMessages: Message[] | null = null

    try {
      for (let attempt = 0; ; attempt++) {
        let received = false
        const pool = currentConnectionPool()
        try {
          const stream = createChatStream({ ...req, messages: history })
          abort = stream.abort
          for await (const event of stream) {
            received = true
            const decoded = decodeChatStreamEvent(event)
            if (!decoded) continue
            if (decoded.kind === "usage") {
              usage = decoded.usage
              setUsage(usage)
              continue
            }
            if (decoded.kind === "done") {
              completed = true
              await pacer.finish()
              break
            }
            if (decoded.kind === "token") {
              pacer.push(decoded.text)
              continue
            }
            if (decoded.kind === "error") pacer.flush()
            applyChatStreamEvent(answer, decoded)
            publish()
            // An error event is the turn's last content: append it, then stop reading.
            if (decoded.kind === "error") {
              unresolvedReason = decoded.message
              break
            }
          }
          break
        } catch (e) {
          // A stream that died before any event is sent again, over a fresh connection
          if (received || attempt >= 2 || !(e instanceof TypeError)) throw e
          leaveConnectionPool(pool)
        }
      }
    } catch (e) {
      // An abort keeps what streamed. A transport failure also says so in the reply, or it would show an empty bubble.
      const aborted = e instanceof Error && e.name === "AbortError"
      unresolvedReason = aborted ? "Cancelled" : errorMessage(e, "Stream failed")
      if (!aborted) {
        pacer.flush()
        applyChatStreamEvent(answer, { kind: "error", message: unresolvedReason })
      }
    } finally {
      pacer.flush()
      settleRunningToolCalls(answer, unresolvedReason)
      publish()
      if (store.get().messages.at(-1) === shown) {
        finalMessages = update((current) => deactivateSentImages([...current.slice(0, -1), { ...answer }], req.img_paths))
      }
      setStreaming(false)
      abort = null
    }

    return completed && finalMessages ? { messages: finalMessages, usage } : null
  }

  return {
    get: store.get,
    subscribe: store.subscribe,
    update,
    reset: (messages = [], usage = NO_USAGE) => store.set({ ...store.get(), messages, usage }),
    send,
    cancel: () => {
      abort?.()
      setStreaming(false)
    },
  }
}
