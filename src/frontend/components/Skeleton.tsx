import { cn } from "@/lib/utils"

interface SkeletonProps {
  className?: string
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        "animate-shimmer rounded bg-gradient-to-r from-zinc-800 via-zinc-700 to-zinc-800",
        className
      )}
    />
  )
}
