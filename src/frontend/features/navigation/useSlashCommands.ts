"use client"

import { useCallback, useMemo } from "react"
import type { CommandMenuItem } from "@/features/navigation/CommandMenu"

const COMMANDS: CommandMenuItem[] = [
  { command: "/memorize", keywords: ["memory", "global", "profile", "project"] },
  { command: "/files-load" },
]

type SlashCommand = { kind: "files-load" } | { kind: "memorize" }

const MEMORIZE = /^\/memorize(?:\s|$)/

/** Read a trimmed message as a slash command, or null when it is ordinary chat text. */
function parseSlashCommand(text: string): SlashCommand | null {
  if (text === "/files-load") return { kind: "files-load" }
  if (MEMORIZE.test(text)) return { kind: "memorize" }
  return null
}

/** The slash-command menu items and one dispatcher shared by the menu and the chat input. */
export function useSlashCommands({
  filesEnabled,
  openFilePicker,
  memorize,
}: {
  filesEnabled: boolean
  openFilePicker: () => void
  memorize: () => Promise<void>
}) {
  const menuItems = useMemo(
    () => COMMANDS.filter((item) => item.command !== "/files-load" || filesEnabled),
    [filesEnabled],
  )

  /** Run `text` if it is a slash command and report whether it was one. `/files-load` without file roots does nothing. */
  const runSlashCommand = useCallback(async (text: string): Promise<boolean> => {
    const command = parseSlashCommand(text)
    if (!command) return false
    if (command.kind === "memorize") await memorize()
    else if (filesEnabled) openFilePicker()
    return true
  }, [filesEnabled, openFilePicker, memorize])

  return { menuItems, runSlashCommand }
}
