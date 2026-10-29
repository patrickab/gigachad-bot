import { openHintGuest, type HintGuest, type HintPeer } from "@gigachad/chat-ui/hints"
import type { FrameMessage, HintMessage, ShellMessage } from "@gigachad/chat-ui/protocol"

/** How long the frame gets to report its clickables before the shell's hints open without them. */
const FRAME_ANSWER_MS = 300

export function isHintMessage(message: FrameMessage): message is Extract<FrameMessage, HintMessage> {
  return message.type.startsWith("hints-")
}

/**
 * The shell's side of click hints spanning the omp frame (`hints-*` messages, `@gigachad/chat-ui/hints`).
 *
 * - `peer`: the frame as guest of a session the shell hosts (`f` in the shell)
 * - `handle`: frame messages, the frame's answer to `peer.open` or a session the frame hosts (`f` in the frame)
 */
export function createFrameHints(post: (message: ShellMessage) => void) {
  let guest: HintGuest | null = null
  let answerOpen: ((count: number) => void) | null = null

  const peer: HintPeer = {
    open: () => {
      const { promise, resolve } = Promise.withResolvers<number>()
      let timer = 0
      const finish = (count: number) => {
        window.clearTimeout(timer)
        answerOpen = null
        resolve(count)
      }
      timer = window.setTimeout(() => finish(0), FRAME_ANSWER_MS)
      answerOpen = finish
      post({ type: "hints-open" })
      return promise
    },
    show: (labels, typed) => post({ type: "hints-show", labels, typed }),
    activate: (index) => post({ type: "hints-activate", index }),
    close: () => post({ type: "hints-close" }),
  }

  const closeGuest = () => {
    guest?.close()
    guest = null
  }
  const handle = (message: HintMessage) => {
    if (message.type === "hints-targets") answerOpen?.(message.count)
    else if (message.type === "hints-open") {
      closeGuest()
      guest = openHintGuest()
      post({ type: "hints-targets", count: guest.count })
    }
    else if (message.type === "hints-show") guest?.show(message.labels, message.typed)
    else if (message.type === "hints-activate") guest?.activate(message.index)
    else closeGuest()
  }

  return { peer, handle }
}
