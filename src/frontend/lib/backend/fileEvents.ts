"use client"

import { subscribeToChanges } from "./syncStream"

/** A file that changed its path: `from` and `to` are absolute paths. */
export interface FileMove {
  from: string
  to: string
}

type FileMoveListener = (move: FileMove) => void

const localMoveListeners = new Set<FileMoveListener>()
const localChangeListeners = new Set<(path: string) => void>()

/** Tell every `onFileMove` listener that this device renamed a file, before any tab follows the new path. */
export function announceFileMove(move: FileMove): void {
  for (const listener of localMoveListeners) listener(move)
}

/** Call `listener` for every file moved by this device (`announceFileMove`) or by another one (a `moved_to` change event). Returns the unsubscribe. */
export function onFileMove(listener: FileMoveListener): () => void {
  localMoveListeners.add(listener)
  const stopRemote = subscribeToChanges((event) => {
    if (event.resource_kind === "document" && event.moved_to) listener({ from: event.resource_key, to: event.moved_to })
  })
  return () => {
    localMoveListeners.delete(listener)
    stopRemote()
  }
}

/** Tell every `onFileChange` listener that this device changed or created the file at `path` through the backend (its own change events never come back). */
export function announceFileChange(path: string): void {
  for (const listener of localChangeListeners) listener(path)
}

/** Call `listener` with the path of every file the predicate (or the exact path) accepts whose content changed: on another device, through the backend, or announced by this device. Returns the unsubscribe. */
export function onFileChange(match: string | ((path: string) => boolean), listener: (path: string) => void): () => void {
  const accepts = typeof match === "string" ? (path: string) => path === match : match
  const local = (path: string) => {
    if (accepts(path)) listener(path)
  }
  localChangeListeners.add(local)
  const stopRemote = subscribeToChanges((event) => {
    if (event.resource_kind === "document" && !event.moved_to && accepts(event.resource_key)) listener(event.resource_key)
  })
  return () => {
    localChangeListeners.delete(local)
    stopRemote()
  }
}
