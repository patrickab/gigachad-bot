"use client"

import { useCallback, useMemo } from "react"
import type { CommandMenuItem } from "@/features/navigation/CommandMenu"
import type { MemorizeScope } from "@/features/memory/useMemoryCommandBar"

const COMMANDS: CommandMenuItem[] = [
  { command: "/memorize", keywords: ["memory", "both"] },
  { command: "/memorize-global", keywords: ["memory", "global", "profile"] },
  { command: "/memorize-project", keywords: ["memory", "project"] },
  { command: "/files-load" },
]

type SlashCommand =
  | { kind: "files-load" }
  | { kind: "memorize"; scope: MemorizeScope }

const MEMORIZE = /^\/memorize(?:-(global|project))?(?:\s|$)/

/** Read a trimmed message as a slash command, or null when it is ordinary chat text. */
function parseSlashCommand(text: string): SlashCommand | null {
  if (text === "/files-load") return { kind: "files-load" }
  const memorize = MEMORIZE.exec(text)
  if (memorize) return { kind: "memorize", scope: memorize[1] === "global" || memorize[1] === "project" ? memorize[1] : null }
  return null
}

/** The slash-command menu items and one dispatcher shared by the menu and the chat input. */
export function useSlashCommands({
  filesEnabled,
  hasProject,
  openFilePicker,
  memorize,
}: {
  filesEnabled: boolean
  hasProject: boolean
  openFilePicker: () => void
  memorize: (scope: MemorizeScope) => Promise<void>
}) {
  const menuItems = useMemo(
    () => COMMANDS.filter((item) =>
      (item.command !== "/files-load" || filesEnabled) &&
      (item.command !== "/memorize-project" || hasProject)),
    [filesEnabled, hasProject],
  )

  /** Run `text` if it is a slash command and report whether it was one. */
  const runSlashCommand = useCallback(async (text: string): Promise<boolean> => {
    const command = parseSlashCommand(text)
    if (!command) return false
    if (command.kind === "files-load") openFilePicker()
    else await memorize(command.scope)
    return true
  }, [openFilePicker, memorize])

  return { menuItems, runSlashCommand }
}
