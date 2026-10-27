import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

/** Small alert paragraph for an error message, nothing when there is none. */
export function ErrorText({ children, className }: { children?: ReactNode; className?: string }) {
  if (!children) return null
  return <p role="alert" className={cn("text-xs text-danger", className)}>{children}</p>
}
