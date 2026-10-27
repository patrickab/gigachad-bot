"use client"

import { useState } from "react"

/** Call `reset(value)` during render whenever `value` changes, the render-time alternative to an effect. */
export function useResetOnChange<T>(value: T, reset: (value: T) => void): void {
  const [previous, setPrevious] = useState(value)
  if (!Object.is(value, previous)) {
    setPrevious(value)
    reset(value)
  }
}

/** Call `reset()` during render each time `open` turns true. */
export function useResetOnOpen(open: boolean, reset: () => void): void {
  useResetOnChange(open, (isOpen) => { if (isOpen) reset() })
}
