import { cn } from "@/lib/utils"

interface SkeletonProps {
  className?: string
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        "animate-shimmer rounded bg-gradient-to-r from-surface-elevated via-surface to-surface-elevated",
        className
      )}
    />
  )
}
