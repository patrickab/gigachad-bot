/**
 * postMessage protocol between gigachad-bot's shell (parent window) and LearnHarness's overlay (`learnharness-overlay/`) in its frame.
 *
 * - The one definition both sides import: the shell from source (only `features/learn`), the overlay as `@gigachad/chat-ui/learn-protocol`
 * - Separate from omp's `protocol.ts`: own channel, so neither frame reacts to the other's traffic
 * - Every message carries `channel: LEARN_EMBED_CHANNEL`, the frame accepts messages from `window.parent` only
 *   (LearnHarness's demo iframes post messages too) and pins the shell origin at `hello`
 * - Versioned: `ready` and `hello` carry `LEARN_PROTOCOL_VERSION`. Drive mode (the shell's model selector and chrome over
 *   upstream's hidden ones) runs only when both sides speak the same version. Otherwise the frame keeps upstream's UI
 *   visible and only the messages of `ANY_VERSION_SHELL_MESSAGE` flow. The frame's bundle changes only when LearnHarness
 *   restarts, so a mismatch means "LearnHarness needs a restart"
 */
export const LEARN_EMBED_CHANNEL = "learn-embed"
export const LEARN_PROTOCOL_VERSION = 3

export type LearnTheme = "light" | "dark"

/** LearnHarness's two session kinds: a teaching session, or practice (upstream's Practice entry, `/chat/new?mode=practice`). */
export type LearnSessionKind = "teach" | "practice"

/**
 * A model as LearnHarness stores it (`ChatSettings.model` is `{provider, id}`), keyed like omp's `ModelRef`.
 * Gigachad's selector string is `modelId`, `provider` is "custom" for every served model.
 */
export type LearnModelRef = { provider: string; modelId: string }

/** One entry of LearnHarness's "Background agents" list. */
export type LearnActivityItem = { id: string; label: string; status: string; sessionId: string | null }

/**
 * What a session row of LearnHarness's hidden sidebar said besides its title (upstream's `sessionMeta`), most urgent first:
 * a context compaction runs, a question waits for the learner, a practice problem waits, a turn runs. Null: idle.
 */
export const LEARN_SESSION_STATUSES = ["compacting", "waiting", "problem", "working"] as const
export type LearnSessionStatus = (typeof LEARN_SESSION_STATUSES)[number]

/**
 * Upstream shortcuts (`KeyAction`) of the hidden sidebar and title bar. Pressed inside the frame they reach the shell as
 * `chrome-key`, whose chrome stands in: its sidebar (⌘B), its learn session filter (⌘⇧F), its rename of the open session (F2).
 */
export const LEARN_CHROME_KEYS = ["toggle-sidebar", "sessions-search", "rename-session"] as const
export type LearnChromeKey = (typeof LEARN_CHROME_KEYS)[number]

/**
 * Upstream UI of the hidden chrome the shell opens inside the frame (`open-panel`), by what it needs:
 * `"session"` panels show for the open session `/chat/<sessionId>`, `"app"` panels anywhere.
 *
 * - `plan`: the title bar's Plan chip, upstream's full-plan popup
 * - `coverage`: the title bar's coverage ring and the session menu's coverage block, with "Extract now"
 * - `activity`: the sidebar's "Background agents" section (per-agent detail, running cost, stop)
 */
export const LEARN_PANEL_SCOPE = {
  plan: "session",
  coverage: "session",
  activity: "app",
} as const satisfies Record<string, "session" | "app">
export type LearnPanel = keyof typeof LEARN_PANEL_SCOPE

function includes<Item extends string>(list: readonly Item[], value: unknown): value is Item {
  return typeof value === "string" && (list as readonly string[]).includes(value)
}

export const isLearnSessionStatus = (value: unknown): value is LearnSessionStatus => includes(LEARN_SESSION_STATUSES, value)
export const isLearnChromeKey = (value: unknown): value is LearnChromeKey => includes(LEARN_CHROME_KEYS, value)
export const isLearnPanel = (value: unknown): value is LearnPanel => typeof value === "string" && Object.hasOwn(LEARN_PANEL_SCOPE, value)

/** Shell → frame. */
export type LearnShellMessage =
  /** First message after `ready`: fixes the shell origin and the theme. A frame at another `version` stays in upstream's look. */
  | { type: "hello"; theme: LearnTheme; version: number }
  | { type: "set-theme"; theme: LearnTheme }
  /** Show LearnHarness session `sessionId` (`/chat/<id>`). */
  | { type: "open-session"; sessionId: string }
  /** Start a new session of `kind` (`/chat/new`, practice: upstream's Practice entry). */
  | { type: "new-session"; kind: LearnSessionKind }
  /** LearnHarness's home (`/`). */
  | { type: "open-home" }
  /** Draft (`sessionId` null): sets the strong tier. Session: only honoured while `model.unavailable`, else dropped. */
  | ({ type: "set-model"; sessionId: string | null } & LearnModelRef)
  | { type: "set-thinking"; sessionId: string | null; level: string }
  /** LearnHarness's settings dialog (the cheap and vision tiers live there). `section` null: upstream's default, kept off hidden sections. */
  | { type: "open-settings"; section: string | null }
  /** Open upstream's `panel` in the frame. `sessionId`: the shown session for a `"session"` panel, null for an `"app"` one. */
  | { type: "open-panel"; panel: LearnPanel; sessionId: string | null }
  /** Close the frame's open modal dialog (`modal`) like Escape inside it: the shell's dimmed chrome was clicked or Escape pressed in the shell. */
  | { type: "close-modal" }

/** Frame → shell. */
export type LearnFrameMessage =
  /** Sent on every load, to any origin: the shell answers with `hello`. */
  | { type: "ready"; version: number }
  /** The open session after every navigation, null outside `/chat/<id>` (and on `/chat/new`). */
  | { type: "view"; sessionId: string | null }
  /**
   * The model pills' state, from every surface that mounts the control (home, `/chat/new`, session).
   * `locked`: session model, read-only. `unavailable`: the model is no longer served, a pick is allowed.
   * `thinkingLevel` null: no model picked.
   */
  | { type: "model"; sessionId: string | null; model: LearnModelRef | null; thinkingLevel: string | null; thinkingLevels: string[]; locked: boolean; unavailable: boolean }
  /** What the proxy serves (`GET /api/models`), so the shell offers only models LearnHarness can run. Repeated on `lh:settings-saved` models. */
  | { type: "models"; served: LearnModelRef[] }
  /** Upstream asked for its model menu (⌘K, the send guard's `openModelPicker`), the shell opens its selector. */
  | { type: "model-open"; sessionId: string | null }
  /** A modal dialog (setup, settings) is open, like omp's `modal`. */
  | { type: "modal"; open: boolean }
  /**
   * What the hidden sidebar row and title bar showed for `sessionId`, on every change: title, kind, phase chip (null for
   * practice), the row's `status` and when the session last changed (ISO time, the row's "5m ago").
   */
  | { type: "session-meta"; sessionId: string; title: string; kind: LearnSessionKind | null; phase: string | null; status: LearnSessionStatus | null; updatedAt: string }
  /** Whether the open session's hidden title bar shows its Plan chip (the `plan` panel has something to show), on every change. */
  | { type: "session-plan"; sessionId: string; available: boolean }
  /** What the hidden "Background agents" section listed, on every change. */
  | { type: "activity"; items: LearnActivityItem[] }
  /** A hidden chrome shortcut pressed inside the frame (`LEARN_CHROME_KEYS`), the shell's chrome acts on it. */
  | { type: "chrome-key"; action: LearnChromeKey }

export type LearnShellMessageType = LearnShellMessage["type"]
export type LearnFrameMessageType = LearnFrameMessage["type"]
export type LearnShellMessageOf<Type extends LearnShellMessageType> = Extract<LearnShellMessage, { type: Type }>
export type LearnFrameMessageOf<Type extends LearnFrameMessageType> = Extract<LearnFrameMessage, { type: Type }>

/**
 * How each frame message is kept, exhaustive: a new frame message fails `tsc` until it is classified here.
 *
 * - `"latest"`: state, the frame keeps the latest one and replays it on every `hello`, the shell keeps the latest one
 * - `"per-session"`: state kept per `sessionId` (the latest for each session), replayed and kept the same way
 * - `false`: an event (or the `ready` handshake), delivered once and never kept
 */
export const IS_STATE = {
  ready: false,
  view: "latest",
  model: "latest",
  models: "latest",
  "model-open": false,
  modal: "latest",
  "session-meta": "per-session",
  "session-plan": "per-session",
  activity: "latest",
  "chrome-key": false,
} as const satisfies Record<LearnFrameMessageType, "latest" | "per-session" | false>

/** Frame messages replayed as state. */
export type LearnStateType = { [Type in LearnFrameMessageType]: (typeof IS_STATE)[Type] extends false ? never : Type }[LearnFrameMessageType]
/** Frame messages delivered as events (everything but state and the `ready` handshake). */
export type LearnEventType = Exclude<LearnFrameMessageType, LearnStateType | "ready">

/**
 * Shell messages each side sends and honours whatever the other side's version, exhaustive like `IS_STATE`: the
 * handshake, the theme and navigation (v1's set). Every other shell message needs drive mode and is dropped without it.
 */
export const ANY_VERSION_SHELL_MESSAGE = {
  hello: true,
  "set-theme": true,
  "open-session": true,
  "new-session": true,
  "open-home": true,
  "set-model": false,
  "set-thinking": false,
  "open-settings": false,
  "open-panel": false,
  "close-modal": false,
} as const satisfies Record<LearnShellMessageType, boolean>

/** Version a `ready` or `hello` without one speaks: the first protocol carried none. */
export const UNVERSIONED_PROTOCOL = 1

export type LearnEnvelope<Message> = Message & { channel: typeof LEARN_EMBED_CHANNEL }
