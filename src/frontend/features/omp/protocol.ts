/**
 * postMessage protocol between this shell and omp-web in its frame.
 *
 * - Mirror of ompweb `custom/embed/protocol.ts`, change both together and bump the version
 * - Every message carries `channel: EMBED_CHANNEL` so unrelated postMessage traffic is ignored
 */
export const EMBED_CHANNEL = "omp-embed"
export const EMBED_PROTOCOL_VERSION = 1

export type ThemeMode = "light" | "dark"

/** Key combination the frame hands back to the shell instead of handling it. `key` is `KeyboardEvent.key`. */
export type KeyChord = { key: string; altKey: boolean; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }

/** Message-only view of an omp session tree entry. */
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

export type ShellMessage =
  /** Sent on every `ready`: the frame learns the shell origin and replays its current state. */
  | { type: "hello"; version: number; forwardKeys: KeyChord[]; theme: ThemeMode }
  | { type: "open-session"; sessionId: string }
  /** Opens the new-session composer in `cwd`, like Ctrl+Alt+N. */
  | { type: "new-session"; cwd: string }
  | { type: "open-settings" }
  | { type: "set-theme"; theme: ThemeMode }
  | { type: "select-leaf"; sessionId: string; leafId: string | null }

export type FrameMessage =
  /** Posted on every page load, the only message sent before `hello` (target origin "*", no data). */
  | { type: "ready"; version: number }
  /** Session shown in the chat (null: new-session screen) and whether full-page Settings covers it. */
  | { type: "view"; sessionId: string | null; settingsOpen: boolean }
  | { type: "branch-tree"; sessionId: string; tree: BranchNode[]; activeLeafId: string | null }
  | ({ type: "key" } & KeyChord)

export type Envelope<Message> = Message & { channel: typeof EMBED_CHANNEL }
