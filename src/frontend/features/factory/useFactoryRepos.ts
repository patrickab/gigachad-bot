import { useEffect, useState } from "react"
import { fetchFactoryRepos, registerFactoryRepo } from "@/lib/backend/api"
import type { FactoryRepo } from "@/lib/domain/types"

function trimSlash(path: string): string {
  return path.replace(/\/+$/, "")
}

/** Repos registered with the factory engine, loaded once while `enabled`; empty while the engine is down. */
export function useFactoryRepos(enabled: boolean): (directory: string) => Promise<FactoryRepo> {
  const [repos, setRepos] = useState<FactoryRepo[]>([])

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    fetchFactoryRepos().then((loaded) => { if (!cancelled) setRepos(loaded) }, () => {})
    return () => { cancelled = true }
  }, [enabled])

  /** The registered repo at *directory*, registering the checkout first when the engine does not know it yet. */
  return async (directory) => {
    const known = repos.find((repo) => trimSlash(repo.path) === trimSlash(directory))
    if (known) return known
    const registered = await registerFactoryRepo(directory)
    setRepos((current) => [...current.filter((repo) => repo.name !== registered.name), registered])
    return registered
  }
}
