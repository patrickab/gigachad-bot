export interface LatexIssue {
  /** Line of the built file, null when the log names none. */
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
