import { createContext, useCallback, useContext, useMemo, useState, type ReactNode, type RefCallback } from "react"
import { createPortal } from "react-dom"

/** Named places in a tab's top bar that the surface underneath renders controls into. */
export type TopBarSlotName = "wideToggle" | "pdf" | "drawing" | "view"

type Slots = Record<TopBarSlotName, HTMLElement | null>

interface TopBarSlotRegistry {
  slots: Slots
  /** Make `element` the `name` slot; the returned cleanup clears it again unless another element took over. */
  register: (name: TopBarSlotName, element: HTMLElement) => () => void
}

const NO_SLOTS: Slots = { wideToggle: null, pdf: null, drawing: null, view: null }

const TopBarSlotContext = createContext<TopBarSlotRegistry>({ slots: NO_SLOTS, register: () => () => {} })

/** A scope of top-bar slots: consumers below see only the slots registered inside it, so a nested provider hides the outer ones. */
export function TopBarSlotProvider({ children }: { children: ReactNode }) {
  const [slots, setSlots] = useState<Slots>(NO_SLOTS)
  const register = useCallback((name: TopBarSlotName, element: HTMLElement) => {
    setSlots((current) => current[name] === element ? current : { ...current, [name]: element })
    return () => setSlots((current) => current[name] === element ? { ...current, [name]: null } : current)
  }, [])
  const value = useMemo(() => ({ slots, register }), [slots, register])
  return <TopBarSlotContext.Provider value={value}>{children}</TopBarSlotContext.Provider>
}

/** Registers the element `children` puts `ref` on as the `name` slot of the nearest provider. */
export function TopBarSlot({ name, children }: { name: TopBarSlotName; children: (ref: RefCallback<HTMLElement>) => ReactNode }) {
  const { register } = useContext(TopBarSlotContext)
  const ref = useCallback((element: HTMLElement | null) => element ? register(name, element) : undefined, [register, name])
  return children(ref)
}

/** The element registered as the `name` slot, null while none is mounted. */
export function useTopBarSlot(name: TopBarSlotName): HTMLElement | null {
  return useContext(TopBarSlotContext).slots[name]
}

/** Whether every one of `names` has a mounted slot, for layouts that differ when their controls live in the top bar. */
export function useHasTopBarSlot(...names: TopBarSlotName[]): boolean {
  const { slots } = useContext(TopBarSlotContext)
  return names.every((name) => slots[name] !== null)
}

/** Renders `children` into the `name` slot, or nothing while that slot is not mounted. */
export function InTopBar({ name, children }: { name: TopBarSlotName; children: ReactNode }) {
  const slot = useTopBarSlot(name)
  return slot ? createPortal(children, slot) : null
}
