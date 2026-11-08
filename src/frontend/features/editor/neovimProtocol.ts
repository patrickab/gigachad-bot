/** The JSON frames between the browser and a backend Neovim session, as `src/backend/routes/neovim.py` documents them. */

/** What the app asks of a running session. */
export type NeovimCommand =
  /** Show the buffer of `path`, loading it from disk when not open yet. A positive `line` puts the cursor there at 0-based byte `column`. */
  | { type: "open_file"; path: string; line?: number; column?: number }
  /** The file changed on disk: the buffer of `path` takes `content`, merged into its unsaved edits. */
  | { type: "set_content"; content: string; path: string }
  /** The file of buffer `from` was renamed to `path`, the buffer keeps its unsaved edits. */
  | { type: "renamed"; from: string; path: string }
  /** Lines below the cursor of the shown buffer. */
  | { type: "insert_lines"; text: string }
  /** A Beamer frame after the frame enclosing the cursor, else below the cursor. */
  | { type: "insert_frame"; text: string }
  /** `path` fills the cursor line's first `example-image` placeholder, else the lines of `reference` go below the cursor. */
  | { type: "insert_image"; path: string; reference: string }
  /** Neovim shows `text` as an error, for failures that have no request. */
  | { type: "message"; text: string }

/** Frames the session hook itself sends to keep the terminal running. */
export type NeovimSessionFrame =
  | { type: "open"; path: string; filename: string; content: string; base_content?: string; ask_keymap: boolean; notebook?: boolean; rows: number; cols: number }
  | { type: "resize"; rows: number; cols: number }
  | { type: "saved"; id: number; content: string }
  | { type: "saved"; id: number; error: string }
  | { type: "background"; light: boolean }
  | { type: "clipboard"; id: number; text: string }
  | { type: "clipboard"; id: number; error: string }

/** What a session reports to the app. */
export type NeovimEvent =
  /** Lines picked in visual mode with the ask keymap, 1-based and inclusive, `text` is those full lines of the file at `path`. */
  | { type: "selection"; path: string; start_line: number; end_line: number; text: string }
  /** Neovim itself (`:bnext`) put the buffer of another open file on screen. Not sent after `open_file`. */
  | { type: "switched"; path: string }
  /** `open_file` could not load the file, the buffer shown did not change. */
  | { type: "open_failed"; path: string; message: string }
  /** The notebook buffer `path` ran the cell holding its 1-based `line` (`<S-CR>`/`<C-CR>`), after the save before it resolved. */
  | { type: "run_cell"; path: string; line: number }
  /** Normal-mode `<leader>v` in a `.tex` buffer asked for the PDF spot of 1-based `line` of the file at `path`. */
  | { type: "forward_search"; path: string; line: number }

/** One handler per event type, each optional. */
export type NeovimEventHandlers = Partial<{ [K in NeovimEvent["type"]]: (event: Extract<NeovimEvent, { type: K }>) => void }>

/** Whether a command waits for the open frame while the socket connects, so a file opened or renamed meanwhile is not lost. */
export const QUEUES_WHILE_CONNECTING: Record<NeovimCommand["type"], boolean> = {
  open_file: true,
  renamed: true,
  set_content: false,
  insert_lines: false,
  insert_frame: false,
  insert_image: false,
  message: false,
}

const APP_EVENTS: Record<NeovimEvent["type"], true> = { selection: true, switched: true, open_failed: true, run_cell: true, forward_search: true }

/** The frame `command` goes out as: `open_file` names no column for column 0 and neither without a line. */
export function wireFrame(command: NeovimCommand): NeovimCommand {
  if (command.type !== "open_file") return command
  const { type, path, line, column } = command
  if (line === undefined) return { type, path }
  return column ? { type, path, line, column } : { type, path, line }
}

/** Hand `message` to its handler in `handlers` when it is an app event. False for any other message. */
export function dispatchNeovimEvent(handlers: NeovimEventHandlers, message: { type: string }): boolean {
  if (!Object.hasOwn(APP_EVENTS, message.type)) return false
  const event = message as NeovimEvent
  const handle = handlers[event.type] as ((event: NeovimEvent) => void) | undefined
  handle?.(event)
  return true
}
