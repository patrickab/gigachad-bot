import {
  ANY_VERSION_SHELL_MESSAGE,
  IS_STATE,
  isLearnPanel,
  LEARN_EMBED_CHANNEL,
  LEARN_PANEL_SCOPE,
  LEARN_PROTOCOL_VERSION,
  UNVERSIONED_PROTOCOL,
  type LearnEnvelope,
  type LearnFrameMessage,
  type LearnSessionKind,
  type LearnShellMessage,
  type LearnShellMessageOf,
  type LearnShellMessageType,
  type LearnTheme,
} from "@gigachad/chat-ui/learn-protocol";

export type ShellHandler<Type extends LearnShellMessageType> = (message: LearnShellMessageOf<Type>) => void;

/** The parts of `window` the channel uses, so tests can drive it without a browser. */
export type ChannelWindow = {
  parent: { postMessage(message: unknown, targetOrigin: string): void };
  addEventListener(type: "message", listener: (event: MessageEvent) => void): void;
  removeEventListener(type: "message", listener: (event: MessageEvent) => void): void;
};

/** Shell message handlers by type, kept apart from the channel so components can register before it opens. */
export type ShellDispatcher = {
  /** Handle one shell message type, returns the unsubscribe. */
  on<Type extends LearnShellMessageType>(type: Type, handler: ShellHandler<Type>): () => void;
  /** Hand `message` to every handler of its type. */
  dispatch(message: LearnShellMessage): void;
};

export type LearnChannel = {
  /** Post to the shell once it said hello. State messages (`IS_STATE`) are also kept and replayed on every `hello`. */
  post(message: LearnFrameMessage): void;
  /** Handle one shell message type on the channel's dispatcher, returns the unsubscribe. */
  on<Type extends LearnShellMessageType>(type: Type, handler: ShellHandler<Type>): () => void;
  /** Stop listening. */
  close(): void;
};

type Fields = Record<string, unknown>;

const isTheme = (value: unknown): value is LearnTheme => value === "light" || value === "dark";
const isId = (value: unknown): value is string => typeof value === "string" && value !== "";
const isOptionalId = (value: unknown): value is string | null => value === null || isId(value);
const isSessionKind = (value: unknown): value is LearnSessionKind => value === "teach" || value === "practice";

/** A `version` field: absent is the unversioned first protocol, anything but a positive integer is malformed. */
function versionOf(value: unknown): number | null {
  if (value === undefined) return UNVERSIONED_PROTOCOL;
  return typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : null;
}

/** One validating parser per shell message type, exhaustive: a new shell message fails `tsc` until it is parsed here. */
const SHELL_PARSERS: { [Type in LearnShellMessageType]: (message: Fields) => LearnShellMessageOf<Type> | null } = {
  hello: ({ theme, version }) => {
    const parsed = versionOf(version);
    return isTheme(theme) && parsed !== null ? { type: "hello", theme, version: parsed } : null;
  },
  "set-theme": ({ theme }) => (isTheme(theme) ? { type: "set-theme", theme } : null),
  "open-session": ({ sessionId }) => (isId(sessionId) ? { type: "open-session", sessionId } : null),
  // A first-protocol shell sends no kind: its new session is a teaching one.
  "new-session": ({ kind }) => (kind === undefined ? { type: "new-session", kind: "teach" } : isSessionKind(kind) ? { type: "new-session", kind } : null),
  "open-home": () => ({ type: "open-home" }),
  "set-model": ({ sessionId, provider, modelId }) =>
    isOptionalId(sessionId) && isId(provider) && isId(modelId) ? { type: "set-model", sessionId, provider, modelId } : null,
  "set-thinking": ({ sessionId, level }) => (isOptionalId(sessionId) && isId(level) ? { type: "set-thinking", sessionId, level } : null),
  "open-settings": ({ section }) => (isOptionalId(section) ? { type: "open-settings", section } : null),
  // A session panel names the session it shows, an app panel none.
  "open-panel": ({ panel, sessionId }) => {
    if (!isLearnPanel(panel)) return null;
    const scoped = LEARN_PANEL_SCOPE[panel] === "session" ? isId(sessionId) : sessionId === null;
    return scoped && isOptionalId(sessionId) ? { type: "open-panel", panel, sessionId } : null;
  },
  "close-modal": () => ({ type: "close-modal" }),
};

function isShellMessageType(type: unknown): type is LearnShellMessageType {
  return typeof type === "string" && Object.hasOwn(SHELL_PARSERS, type);
}

/** A shell message on the learn channel with the fields its type needs (and only those), null for anything else. */
export function parseShellMessage(data: unknown): LearnShellMessage | null {
  if (typeof data !== "object" || data === null) return null;
  const message = data as Fields;
  if (message.channel !== LEARN_EMBED_CHANNEL || !isShellMessageType(message.type)) return null;
  return SHELL_PARSERS[message.type](message);
}

export function createShellDispatcher(): ShellDispatcher {
  const handlersByType = new Map<LearnShellMessageType, Set<(message: LearnShellMessage) => void>>();
  return {
    on(type, handler) {
      const handlers = handlersByType.get(type) ?? new Set();
      handlersByType.set(type, handlers);
      const untypedHandler = handler as (message: LearnShellMessage) => void;
      handlers.add(untypedHandler);
      return () => handlers.delete(untypedHandler);
    },
    dispatch(message) {
      handlersByType.get(message.type)?.forEach((handler) => handler(message));
    },
  };
}

/** Where a state message is kept: one slot per type, or per session for `"per-session"` state. */
function stateKey(message: LearnFrameMessage): string | null {
  const kind = IS_STATE[message.type];
  if (kind === false) return null;
  if (kind === "per-session" && "sessionId" in message) return `${message.type}:${message.sessionId}`;
  return message.type;
}

/**
 * Open the frame's side of the learn embed protocol: listen for the shell and announce the frame with `ready` and its version.
 *
 * - Accepts messages from `window.parent` on `LEARN_EMBED_CHANNEL` only: LearnHarness's demo iframes post messages too
 * - Drops malformed messages (`parseShellMessage`), handlers only see the protocol's shapes
 * - `hello` from one of `shellOrigins` (the configured frame ancestors) pins that origin, every later message must come
 *   from it and every frame message goes only to it. With no `shellOrigins` no shell is ever trusted
 * - A shell at another protocol version gets only `ANY_VERSION_SHELL_MESSAGE` through (theme, navigation), drive
 *   messages are dropped until a `hello` at `LEARN_PROTOCOL_VERSION` arrives
 */
export function openLearnChannel(
  win: ChannelWindow,
  shellOrigins: readonly string[],
  dispatcher: ShellDispatcher = createShellDispatcher(),
): LearnChannel {
  let shellOrigin: string | null = null;
  let shellVersion: number | null = null;
  const latestState = new Map<string, LearnFrameMessage>();

  const send = (message: LearnFrameMessage, targetOrigin: string) => {
    const envelope: LearnEnvelope<LearnFrameMessage> = { ...message, channel: LEARN_EMBED_CHANNEL };
    win.parent.postMessage(envelope, targetOrigin);
  };

  const handleMessage = (event: MessageEvent) => {
    if (event.source !== win.parent) return;
    const message = parseShellMessage(event.data);
    if (!message) return;
    if (message.type === "hello" && shellOrigins.includes(event.origin)) {
      shellOrigin = event.origin;
      shellVersion = message.version;
    }
    if (event.origin !== shellOrigin) return;
    if (!ANY_VERSION_SHELL_MESSAGE[message.type] && shellVersion !== LEARN_PROTOCOL_VERSION) return;
    dispatcher.dispatch(message);
    if (message.type === "hello") latestState.forEach((state) => send(state, event.origin));
  };

  win.addEventListener("message", handleMessage);
  send({ type: "ready", version: LEARN_PROTOCOL_VERSION }, "*");

  return {
    post(message) {
      const key = stateKey(message);
      if (key !== null) latestState.set(key, message);
      if (shellOrigin) send(message, shellOrigin);
    },
    on: dispatcher.on,
    close() {
      win.removeEventListener("message", handleMessage);
    },
  };
}
