/**
 * The canvas the app background draws on, for windows that show the same picture (`AmbientCopy`).
 *
 * The background owns it (`AmbientBackdrop` sets it, null while it has none), readers subscribe, so a replaced canvas
 * after a lost graphics context reaches them too.
 */
let canvas: HTMLCanvasElement | null = null
const listeners = new Set<() => void>()

export function setAmbientCanvas(next: HTMLCanvasElement | null): void {
  canvas = next
  listeners.forEach((listener) => listener())
}

export function getAmbientCanvas(): HTMLCanvasElement | null {
  return canvas
}

export function subscribeAmbientCanvas(listener: () => void): () => void {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}
