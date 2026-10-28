/**
 * postMessage protocol between gigachad-bot's shell (parent window) and omp-web's overlay (`ompweb-overlay/`) in its frame.
 *
 * - The one definition both sides import: the shell from source, the overlay as `@gigachad/chat-ui/protocol`
 * - Bump `EMBED_PROTOCOL_VERSION` with every message change, a running frame may still be an older build
 * - Every message carries `channel: EMBED_CHANNEL` so unrelated postMessage traffic is ignored
 */
export const EMBED_CHANNEL = "omp-embed"
export const EMBED_PROTOCOL_VERSION = 8

export type ThemeMode = "light" | "dark"

/** Key combination the frame hands back to the shell instead of handling it, e.g. Alt+T for a new tab. `key` is `KeyboardEvent.key`. */
export type KeyChord = { key: string; altKey: boolean; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }

/** Message-only view of an omp session tree entry, omp-web's `SessionTreeNode` (`lib/types.ts`). */
export type BranchNode = {
  id: string
  /** Entry type from the session file (`message`, `model_change`, `compaction`, ...). */
  type: string
  role?: "user" | "assistant"
  /** First 80 characters of the message text, empty for non-message entries. */
  preview: string
  label?: string
  children: BranchNode[]
}

/** omp model id, as omp-web's `ChatInput` `model` prop holds it. */
export type ModelRef = { provider: string; modelId: string }

/** Base64 image without data-URL prefix, as omp's `prompt` command takes it. */
export type ComposerImage = { data: string; mimeType: string }

/**
 * One autocomplete row: an omp command (`/name`, extensions, prompts and skills only) or a file under the session cwd (`@path`).
 * Applying it replaces `text.slice(from, to)` with `insert` and puts the caret at `from + caretOffset`.
 */
export type Completion = {
  kind: "extension" | "prompt" | "skill" | "file" | "directory"
  label: string
  detail?: string
  from: number
  to: number
  insert: string
  caretOffset: number
}

export type ShellMessage =
  /** Sent on every `ready`: the frame learns the shell origin and replays its current state. */
  | { type: "hello"; version: number; forwardKeys: KeyChord[]; theme: ThemeMode }
  | { type: "open-session"; sessionId: string }
  /** Opens the new-session composer in `cwd`, like Ctrl+Alt+N. */
  | { type: "new-session"; cwd: string }
  | { type: "open-settings" }
  | { type: "set-theme"; theme: ThemeMode }
  | { type: "select-leaf"; sessionId: string; leafId: string | null }
  /** Pixels of shell chrome laid over the frame's top (top bar) and bottom (composer). The transcript scrolls beneath them. Sent after `hello` and on every change. */
  | { type: "insets"; top: number; bottom: number }
  /** Close the frame's open modal dialog (`modal`) like Escape inside it: the shell's backdrop was clicked or Escape pressed in the shell. */
  | { type: "close-modal" }
  /**
   * The shell's composer replaces omp-web's hidden one. Each names the session it was typed for (null: new-session screen), a frame showing another one ignores it.
   * `composer-send` submits like omp-web's composer: builtins (`/btw`), then a prompt, a follow-up while the agent is busy.
   */
  | { type: "composer-send"; sessionId: string | null; text: string; images: ComposerImage[] }
  | { type: "composer-abort"; sessionId: string | null }
  | ({ type: "composer-set-model"; sessionId: string | null } & ModelRef)
  /** One of the composer's `thinkingLevels`. */
  | { type: "composer-set-thinking"; sessionId: string | null; level: string }
  /** Completions for the draft at `caret`, answered by `composer-completions` with the same `requestId`. */
  | { type: "composer-complete"; sessionId: string | null; requestId: number; text: string; caret: number }

export type FrameMessage =
  /** Posted on every page load, the only message sent before `hello` (target origin "*", no data). */
  | { type: "ready"; version: number }
  /** Session shown in the chat (null: new-session screen) and whether full-page Settings covers it. */
  | { type: "view"; sessionId: string | null; settingsOpen: boolean }
  | { type: "branch-tree"; sessionId: string; tree: BranchNode[]; activeLeafId: string | null }
  | ({ type: "key" } & KeyChord)
  /**
   * Composer state for the shell's composer. `sessionId` is the shown session, null on the new-session screen.
   * `thinkingLevels` is what omp-web's reasoning menu offers (`auto`, `off`, ... `max`), empty when it shows none. `thinkingLevel` is the selected one.
   * `contextTokens` is the context omp reports in use (the count behind omp-web's context ring), null before the first report.
   */
  | { type: "composer"; sessionId: string | null; streaming: boolean; model: ModelRef | null; thinkingLevel: string; thinkingLevels: string[]; contextTokens: number | null }
  /**
   * Text and images omp handed back (an abort withdrew queued messages, the queue panel's Edit), for the shell's composer.
   * Merge like omp-web's `mergeRecoveredText` (`lib/draft-store.ts`): `replace` set and the draft starts with `replace.lead` -> that lead becomes `text`, else `replace.fallback` (or `text`) goes before the draft.
   */
  | { type: "composer-restore"; sessionId: string | null; text: string; replace: { lead: string; fallback: string } | null; images: ComposerImage[] }
  /** Answer to `composer-complete`, empty when nothing matches or the lookup failed. */
  | { type: "composer-completions"; sessionId: string | null; requestId: number; items: Completion[] }
  /** The user clicked a link to a local file, the shell shows it instead of omp-web's file panel. `path` is absolute. */
  | { type: "open-file"; path: string }
  /** Whether a modal dialog (e.g. a subagent transcript) is open. The shell dims and blurs its own chrome around the frame alike. */
  | { type: "modal"; open: boolean }

export type Envelope<Message> = Message & { channel: typeof EMBED_CHANNEL }
