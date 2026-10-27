import { EMBED_CHANNEL, EMBED_PROTOCOL_VERSION, type Envelope, type FrameMessage, type ShellMessage } from "@gigachad/chat-ui/protocol";

type ShellMessageType = ShellMessage["type"];
type ShellHandler<Type extends ShellMessageType> = (message: Extract<ShellMessage, { type: Type }>) => void;

/** Frame messages that describe current state: kept and replayed on every `hello`. */
const IS_STATE_MESSAGE: Record<FrameMessage["type"], boolean> = {
  ready: false, view: true, "branch-tree": true, key: false, composer: true, "composer-restore": false, "composer-completions": false, "open-file": false,
};

let shellOrigin: string | null = null;
let isListening = false;
const latestStateByType = new Map<FrameMessage["type"], FrameMessage>();
const handlersByType = new Map<ShellMessageType, Set<(message: ShellMessage) => void>>();

function isFromShell(event: MessageEvent): boolean {
  return event.source === window.parent && event.data?.channel === EMBED_CHANNEL;
}

function send(message: FrameMessage, targetOrigin: string) {
  const envelope: Envelope<FrameMessage> = { ...message, channel: EMBED_CHANNEL };
  window.parent.postMessage(envelope, targetOrigin);
}

/**
 * Route shell messages to the registered handlers.
 *
 * - Trusts the parent window only: CSP `frame-ancestors` (`custom/framing.ts`) already limits who can frame the app
 * - `hello` fixes the shell origin, every later message must come from it
 */
function handleMessage(event: MessageEvent) {
  if (!isFromShell(event)) return;
  const message = event.data as Envelope<ShellMessage>;
  if (message.type === "hello") shellOrigin = event.origin;
  if (event.origin !== shellOrigin) return;
  handlersByType.get(message.type)?.forEach((handler) => handler(message));
  if (message.type === "hello") latestStateByType.forEach((state) => send(state, event.origin));
}

/** Listen for the shell and announce the frame, idempotent. */
export function startEmbedChannel() {
  if (isListening) return;
  isListening = true;
  window.addEventListener("message", handleMessage);
  send({ type: "ready", version: EMBED_PROTOCOL_VERSION }, "*");
}

/** Post to the shell once it said hello. State messages are also kept for the next `hello`. */
export function postToShell(message: FrameMessage) {
  if (IS_STATE_MESSAGE[message.type]) latestStateByType.set(message.type, message);
  if (shellOrigin) send(message, shellOrigin);
}

/** Whether the shell said hello, so `postToShell` reaches it. */
export function isShellConnected(): boolean {
  return shellOrigin !== null;
}

/** Handle one shell message type, returns the unsubscribe. */
export function onShellMessage<Type extends ShellMessageType>(type: Type, handler: ShellHandler<Type>): () => void {
  const handlers = handlersByType.get(type) ?? new Set();
  handlersByType.set(type, handlers);
  const untypedHandler = handler as (message: ShellMessage) => void;
  handlers.add(untypedHandler);
  return () => handlers.delete(untypedHandler);
}
