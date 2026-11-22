import { useState } from "react"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { callFactory } from "@/lib/backend/api"
import type { FactoryRepoAdd } from "@/lib/domain/factory"

function trimSlash(path: string): string {
  return path.replace(/\/+$/, "")
}

async function listRepos(): Promise<string[]> {
  return (await callFactory<{ repos: string[] }>("repos.list")).repos
}

/** Paths of the repos registered with the factory, loaded once while `enabled`; empty while the backend is down. */
export function useFactoryRepos(enabled: boolean): (directory: string) => Promise<string> {
  const [repos, setRepos] = useState<string[]>([])

  useLoadWhile(enabled, listRepos, setRepos)

  /** The registered repo at *directory*, registering the checkout first when the factory does not know it yet. */
  return async (directory) => {
    const known = repos.find((repo) => trimSlash(repo) === trimSlash(directory))
    if (known) return known
    const added = await callFactory<FactoryRepoAdd>("repos.add", { repo: directory })
    setRepos(added.repos)
    return added.repo
  }
}
