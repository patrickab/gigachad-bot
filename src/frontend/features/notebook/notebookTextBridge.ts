/**
 * The notebook as one plain text (percent format, `notebookText.ts`) for an editor outside the notebook, Neovim.
 *
 * - The Jupyter model stays the single source of truth, the text is its projection without outputs
 * - `subscribe` reports every change of the projection the bridge did not hand out itself (disk follow, CodeMirror, toolbar)
 */
export interface NotebookTextBridge {
  /** Extension of the projection's buffer from `metadata.language_info.file_extension` when it matches `^\.[A-Za-z0-9]{1,8}$`, else `.py`. Fixed for the bridge's life. */
  readonly extension: string
  /** The projection of the model now. */
  text(): string
  /**
   * Apply an edited projection to the model (cells matched so outputs stay), persist at once, resolve with the projection of the result.
   * Rejects with "The notebook changed, write again" when the model's projection differs from the last text handed out (the listeners get the new text at once),
   * and when persisting fails.
   */
  save(text: string): Promise<string>
  /** Run the cell holding the 1-based `line` of the last text `save` resolved with, make it active and scroll it into view. A markdown cell is rendered. */
  runAt(line: number): void
  /** Call `listener` with the new projection whenever it differs from the last one handed out. Returns the unsubscribe. */
  subscribe(listener: (text: string) => void): () => void
}
