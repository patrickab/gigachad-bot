import { useContext, useEffect } from "react"
import { TabContext } from "@/contexts/TabContext"

const directories = new Map<string, string>()

/** The working directory of tab `tabId`: the folder of the project its surface shows, undefined when it shows none. Read when needed, never rendered from. */
export function tabWorkingDirectory(tabId: string): string | undefined {
  return directories.get(tabId)
}

/**
 * Make `directory` the enclosing tab's working directory while the calling surface is mounted (the Pi session's cwd, a factory task's worktree, a workbench file's folder).
 *
 * - Null or empty claims nothing. Outside a tab it does nothing
 */
export function useTabWorkingDirectory(directory: string | null | undefined): void {
  // eslint-disable-next-line no-restricted-syntax -- chrome-blind on purpose: only a tab claims a working directory, never the app's chrome
  const tabId = useContext(TabContext)?.tab.id
  useEffect(() => {
    if (!tabId || !directory) return
    directories.set(tabId, directory)
    return () => {
      if (directories.get(tabId) === directory) directories.delete(tabId)
    }
  }, [tabId, directory])
}
