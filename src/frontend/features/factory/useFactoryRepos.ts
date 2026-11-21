import { useState } from "react"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { listFactoryRepos, openFactoryRepo } from "@/lib/backend/api"
import type { FactoryRepo } from "@/lib/domain/factory"

function trimSlash(path: string): string {
  return path.replace(/\/+$/, "")
}

/** Repos registered with the factory, loaded once while `enabled`; empty while the backend is down. */
export function useFactoryRepos(enabled: boolean): (directory: string) => Promise<FactoryRepo> {
  const [repos, setRepos] = useState<FactoryRepo[]>([])

  useLoadWhile(enabled, listFactoryRepos, setRepos)

  /** The registered repo at *directory*, registering the checkout first when the factory does not know it yet. */
  return async (directory) => {
    const known = repos.find((repo) => trimSlash(repo.path) === trimSlash(directory))
    if (known) return known
    const registered = await openFactoryRepo(directory)
    setRepos((current) => [...current.filter((repo) => repo.path !== registered.path), registered])
    return registered
  }
}
