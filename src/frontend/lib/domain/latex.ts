export interface LatexIssue {
  /** Absolute path of the file the error is in, null when the log names none. */
  file: string | null
  /** Line in `file`, null when the log names none. */
  line: number | null
  message: string
}

export interface LatexBuild {
  ok: boolean
  /** The newest PDF on disk, from an earlier build when this one failed, null when none exists. */
  pdf_path: string | null
  errors: LatexIssue[]
  log_tail: string
  duration_ms: number
}

/**
 * The LaTeX project a file belongs to.
 *
 * - `folder` is the project folder (the root's folder), the chat and the rebuild watch are keyed by it
 * - `root` is the `.tex` that builds, null when no rule found one
 * - `candidates` are the root `.tex` files a picker offers when `root` is null (empty for non-`.tex` files)
 */
export interface TexProject {
  folder: string
  root: string | null
  candidates: string[]
}

/** A pasted image stored on disk: its absolute `path`, the `reference` line to insert into the open file and the `changed` absolute paths of other files the paste rewrote (the root when `graphicx` was added). */
export interface StoredImage {
  path: string
  reference: string
  changed: string[]
}
