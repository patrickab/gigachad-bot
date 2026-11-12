"use client"

// Subscribe a component to shared resources and load them once for every consumer.
// Consumes a `Resource` or `ResourceFamily` from `lib/resource`. Provides a resource's value, through `useResourceState` its
// first read's failure, and through `useResourceFamily` the values of several members of a family.
// A first read with a label runs as a startup read, so a failure reaches the shared banner with its retry and 15 second bound.
// A null label reads quietly: a failure leaves the value unloaded until `enabled` turns on again.

import { useEffect, useMemo, useSyncExternalStore } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { syncCursorReady } from "@/lib/backend/syncStream"
import type { Resource, ResourceFamily } from "@/lib/resource"

// The read stores its result in the resource, the subscription delivers it.
const ignore = () => {}
const noValue = () => undefined
const NO_VALUES: Readonly<Record<string, never>> = {}
const noValues = () => NO_VALUES

/** Run `load` once while `enabled`: as a startup read under `label`, or quietly for a null label. Keep `load` stable. */
function useFirstRead(label: string | null, load: (signal: AbortSignal) => Promise<unknown>, enabled: boolean): string | null {
  const { error } = useStartupRead(label ?? "", load, ignore, enabled && label !== null)
  const quiet = enabled && label === null
  useEffect(() => {
    if (!quiet) return
    const controller = new AbortController()
    // Like a startup read, read once the change feed holds its head, so no later change is missed.
    syncCursorReady().then(() => load(controller.signal)).catch(ignore)
    return () => controller.abort()
  }, [quiet, load])
  return error
}

/** `resource`'s value, undefined until loaded, and why its first read failed. Loads while `enabled` and still unloaded. */
export function useResourceState<T>(label: string | null, resource: Resource<T>, enabled = true): { value: T | undefined; error: string | null } {
  const value = useSyncExternalStore(resource.subscribe, resource.get, noValue)
  const error = useFirstRead(label, resource.load, enabled && value === undefined)
  return { value, error: value === undefined ? error : null }
}

/** `resource`'s value, undefined until loaded. Loads while `enabled` and still unloaded. */
export function useResource<T>(label: string | null, resource: Resource<T>, enabled = true): T | undefined {
  return useResourceState(label, resource, enabled).value
}

/** The members of `family` under `keys` as one store whose snapshot changes only when a member's value does. */
function familyStore<T>(family: ResourceFamily<T>, keys: string[]) {
  const members = keys.map((key) => [key, family(key)] as const)
  let seen: (T | undefined)[] = members.map(() => undefined)
  let snapshot: Readonly<Record<string, T>> = NO_VALUES
  return {
    subscribe: (listener: () => void) => {
      const stops = members.map(([, member]) => member.subscribe(listener))
      return () => stops.forEach((stop) => stop())
    },
    get: () => {
      const current = members.map(([, member]) => member.get())
      if (current.some((value, index) => value !== seen[index])) {
        seen = current
        snapshot = Object.fromEntries(members.flatMap(([key], index) => current[index] === undefined ? [] : [[key, current[index]]]))
      }
      return snapshot
    },
    load: (signal: AbortSignal) => Promise.all(members.filter(([, member]) => member.get() === undefined).map(([, member]) => member.load(signal))),
  }
}

/** Values of `family`'s members under `keys`, the loaded ones only, the same object while none changed. Loads while `enabled` and one is unloaded. */
export function useResourceFamily<T>(label: string | null, family: ResourceFamily<T>, keys: readonly string[], enabled = true): Readonly<Record<string, T>> {
  const keysId = JSON.stringify(keys)
  const store = useMemo(() => familyStore(family, JSON.parse(keysId) as string[]), [family, keysId])
  const values = useSyncExternalStore(store.subscribe, store.get, noValues)
  useFirstRead(label, store.load, enabled && keys.some((key) => !(key in values)))
  return values
}
