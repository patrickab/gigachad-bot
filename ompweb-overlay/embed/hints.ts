import { installClickHints, openHintGuest, type HintGuest, type HintPeer } from "@gigachad/chat-ui/hints";
import { isShellConnected, onShellMessage, postToShell } from "./channel";
import { isUnderShellChrome } from "./insets";

/** How long the shell gets to report its clickables before the frame's hints open without them. */
const SHELL_ANSWER_MS = 300;

/** The shell as the guest of a session the frame hosts (`hints-*` messages). */
function shellPeer(): HintPeer | null {
  if (!isShellConnected()) return null;
  return {
    open: () => {
      const { promise, resolve } = Promise.withResolvers<number>();
      let timer = 0;
      let stop = () => {};
      const finish = (count: number) => {
        stop();
        window.clearTimeout(timer);
        resolve(count);
      };
      stop = onShellMessage("hints-targets", (message) => finish(message.count));
      timer = window.setTimeout(() => finish(0), SHELL_ANSWER_MS);
      postToShell({ type: "hints-open" });
      return promise;
    },
    show: (labels, typed) => postToShell({ type: "hints-show", labels, typed }),
    activate: (index) => postToShell({ type: "hints-activate", index }),
    close: () => postToShell({ type: "hints-close" }),
  };
}

/**
 * Click hints (`f`) spanning the frame and the shell around it. Returns the stop.
 *
 * - `f` in the frame: the frame hosts, the shell's clickables join (top bar, composer, sidebar)
 * - `f` in the shell: the frame joins as guest
 * - Frame content under the shell's top bar and composer gets no label
 */
export function installEmbedHints(): () => void {
  let guest: HintGuest | null = null;
  const closeGuest = () => {
    guest?.close();
    guest = null;
  };
  const stops = [
    installClickHints({ peer: shellPeer, isCovered: isUnderShellChrome }),
    onShellMessage("hints-open", () => {
      closeGuest();
      guest = openHintGuest(isUnderShellChrome);
      postToShell({ type: "hints-targets", count: guest.count });
    }),
    onShellMessage("hints-show", (message) => guest?.show(message.labels, message.typed)),
    onShellMessage("hints-activate", (message) => guest?.activate(message.index)),
    onShellMessage("hints-close", closeGuest),
  ];
  return () => {
    stops.forEach((stop) => stop());
    closeGuest();
  };
}
