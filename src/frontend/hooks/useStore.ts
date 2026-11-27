"use client"

import { useSyncExternalStore } from "react"
import type { Store } from "@/lib/store"

/** `select(value)` of `store`, re-rendering when it changes. `select` returns a primitive or a reference the value holds, never a new object. */
export function useStore<T, S = T>(store: Pick<Store<T>, "get" | "subscribe">, select: (value: T) => S = (value) => value as unknown as S): S {
  const read = () => select(store.get())
  return useSyncExternalStore(store.subscribe, read, read)
}
