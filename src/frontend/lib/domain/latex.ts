import { baseName } from "@/lib/paths"

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

/** The source spot SyncTeX finds for a spot in the PDF: the absolute `path` of the source file and its 1-based `line`. */
export interface SynctexSource {
  path: string
  line: number
}

/** The source spot SyncTeX finds for a double-clicked spot in the PDF, with the 0-based UTF-8 byte `column` of the clicked character in its line (0 when unknown), what Neovim's cursor takes. */
export interface SynctexClickedSource extends SynctexSource {
  column: number
}

/** The PDF spot SyncTeX finds for a source line: 1-based `page`, `x`, `y` in PDF points from its top-left corner and the line's `words` to find on the page. */
export interface SynctexView {
  page: number
  x: number
  y: number
  words: string[]
}

/** A pasted image stored on disk: its absolute `path`, its `placeholder_path` (image folder and file name) relative to the folder it is stored under, the `reference` line to insert into the open file and the `changed` absolute paths of other files the paste rewrote (the root when `graphicx` was added). */
export interface StoredImage {
  path: string
  placeholder_path: string
  reference: string
  changed: string[]
}

/** A template set a new presentation can start from: `name` goes to the create request, `label` is what the file type choice shows. */
export interface TemplateSetInfo {
  name: string
  label: string
}

/** One slide primitive of a template set: `id` is `<group>/<name>`, `group_label` the group's heading, `summary` a one-line plain-text description, `text` the frame to insert, `preview` the version of its preview image, null without one. */
export interface LatexPrimitive {
  id: string
  group: string
  group_label: string
  label: string
  description: string
  summary: string
  text: string
  preview: number | null
}

/** The template set of a LaTeX root: `theme` is null when the root is not a presentation, then `primitives` is empty. */
export interface LatexTemplates {
  theme: string | null
  primitives: LatexPrimitive[]
}

/** The file name the PDF of the LaTeX `root` is saved under: the root's name with `.pdf` for `.tex`. */
export function pdfFileNameOf(root: string): string {
  return `${baseName(root).replace(/\.tex$/i, "")}.pdf`
}
