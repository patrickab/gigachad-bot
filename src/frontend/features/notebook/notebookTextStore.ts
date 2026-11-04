/**
 * The notebook as one plain text (percent format, `notebookText.ts`) for Neovim, held per Neovim session.
 *
 * - The Jupyter model stays the single source of truth, the text is its projection without outputs
 * - The model is watched only while it is bound and a follower listens
 */
import { notebookOf, type SharedNotebookLike } from "./notebookSync"
import { applyNotebookText, cellAtLine, modelText } from "./notebookText"

/** Quiet time after a model change before followers get the new text. */
export const NOTEBOOK_TEXT_DELAY_MS = 150

/** A buffer extension the backend accepts. */
const EXTENSION = /^\.[A-Za-z0-9]{1,8}$/

export interface NotebookTextState {
  /** `metadata.language_info.file_extension` when it matches /^\.[A-Za-z0-9]{1,8}$/, else ".py". Fixed while bound. */
  readonly extension: string
  /** The text last handed to Neovim: at bind, by a handout to followers, by `save`. */
  readonly text: string
}

export interface NotebookTextStore {
  /** React (`useSyncExternalStore`): called after every change of `state()`. */
  subscribe(onChange: () => void): () => void
  /** Null while no model is bound. The same object until it changes. */
  state(): NotebookTextState | null
  /** Bind the notebook's model. `persist` writes it at once, false when that failed. `run` runs the cell at an index. Returns the unbind. */
  bind(model: SharedNotebookLike, persist: () => Promise<boolean>, run: (index: number) => void): () => void
  /** Hand every change of the model's text to `push`, debounced by NOTEBOOK_TEXT_DELAY_MS. Returns the stop. */
  follow(push: (text: string) => void): () => void
  /** Apply Neovim's text to the model, persist, resolve with the stored text. */
  save(text: string): Promise<string>
  /** Run the cell holding the 1-based `line` of the text the last `save` resolved with. */
  runAt(line: number): void
}

interface Binding {
  model: SharedNotebookLike
  persist: () => Promise<boolean>
  run: (index: number) => void
  state: NotebookTextState
}

function extensionOf(model: SharedNotebookLike): string {
  const info = notebookOf(model).metadata.language_info
  const declared = typeof info === "object" && info && !Array.isArray(info) ? info.file_extension : undefined
  return typeof declared === "string" && EXTENSION.test(declared) ? declared : ".py"
}

export function createNotebookTextStore(): NotebookTextStore {
  const subscribers = new Set<() => void>()
  const followers = new Set<(text: string) => void>()
  let bound: Binding | null = null
  let saved: string | null = null
  let watched: SharedNotebookLike | null = null
  let timer: ReturnType<typeof setTimeout> | undefined

  const notify = () => {
    for (const onChange of [...subscribers]) onChange()
  }
  const setText = (binding: Binding, text: string) => {
    if (binding.state.text === text) return
    binding.state = { ...binding.state, text }
    notify()
  }
  // Hand the model's text to the followers when it is not the one they last got
  const handOut = () => {
    clearTimeout(timer)
    if (!bound) return
    const text = modelText(bound.model)
    if (text === bound.state.text) return
    setText(bound, text)
    for (const push of [...followers]) push(text)
  }
  const onUpdate = () => {
    clearTimeout(timer)
    timer = setTimeout(handOut, NOTEBOOK_TEXT_DELAY_MS)
  }
  // Watch the bound model's updates exactly while someone follows
  const watch = () => {
    const target = bound && followers.size > 0 ? bound.model : null
    if (target === watched) return
    clearTimeout(timer)
    watched?.ydoc.off("update", onUpdate)
    target?.ydoc.on("update", onUpdate)
    watched = target
  }

  return {
    subscribe(onChange) {
      subscribers.add(onChange)
      return () => { subscribers.delete(onChange) }
    },
    state: () => bound?.state ?? null,
    bind(model, persist, run) {
      const binding = { model, persist, run, state: { extension: extensionOf(model), text: modelText(model) } }
      bound = binding
      saved = null
      watch()
      notify()
      return () => {
        if (bound !== binding) return
        bound = null
        saved = null
        clearTimeout(timer)
        watch()
        notify()
      }
    },
    follow(push) {
      followers.add(push)
      watch()
      handOut()
      return () => {
        followers.delete(push)
        watch()
      }
    },
    async save(edited) {
      const binding = bound
      if (!binding) throw new Error("The notebook is not loaded")
      // The model moved on since the text was handed out: applying it would silently revert that change
      if (modelText(binding.model) !== binding.state.text) {
        handOut()
        throw new Error("The notebook changed, write again")
      }
      applyNotebookText(binding.model, edited)
      // Before the persist: the update it caused is no news to Neovim, which saved it
      setText(binding, modelText(binding.model))
      if (!(await binding.persist())) throw new Error("Saving the notebook failed")
      const stored = modelText(binding.model)
      if (bound === binding) {
        saved = stored
        setText(binding, stored)
      }
      return stored
    },
    runAt(line) {
      if (saved === null) return
      const index = cellAtLine(saved, line)
      if (index !== null) bound?.run(index)
    },
  }
}
