"use client"

import { useContext, type Context } from "react"

/** Value of `context`, throwing `message` when used outside its provider. */
export function useRequiredContext<T>(context: Context<T | null>, message: string): T {
  const value = useContext(context)
  if (value === null) throw new Error(message)
  return value
}
