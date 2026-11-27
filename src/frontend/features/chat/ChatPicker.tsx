"use client"

import { useState } from "react"
import { FloatingPicker, useAltKeyPicker, type PickerGroup } from "@/components/ui/CommandPicker"
import { ErrorText } from "@/components/ui/ErrorText"
import { useProject } from "@/contexts/ProjectContext"
import { useNowMs } from "@/hooks/useNowMs"
import type { ChatListing, ChatSearchMatch } from "@/lib/backend/api"
import type { ChatRevealHit } from "./ChatContainer"
import type { ProjectListItem } from "@/lib/domain/projects"
import { cn, compactElapsed } from "@/lib/utils"
import { QUERY_MAX_LENGTH, useAllChats, useChatSearch } from "./useChatPicker"

const GENERAL = "General"
const ROW_TEXT_WIDTH = "max-w-[16rem]"

/** A row: a chat, or one hit in it (`first`: the chat's first hit row, the one that names it). */
interface Entry { chat: ChatListing; match?: ChatSearchMatch; first?: boolean }

const entryKey = ({ chat, match }: Entry) => (match ? `${chat.historyFile}:${match.qaIndex}:${match.role}:${match.ordinal}` : chat.historyFile)

type OpenChat = (historyFile: string, qaIndex?: number, hit?: ChatRevealHit) => void

const newestFirst = (left: ChatListing, right: ChatListing) => right.updatedAt.localeCompare(left.updatedAt)

/** One group per project (chats without one in "General"), groups ordered by their newest chat, chats newest first. */
export function groupByProject(chats: readonly ChatListing[], projects: readonly ProjectListItem[]): PickerGroup<Entry>[] {
  const groups = new Map<string, PickerGroup<Entry>>()
  for (const chat of [...chats].sort(newestFirst)) {
    const key = chat.projectSlug ?? ""
    let group = groups.get(key)
    if (!group) {
      const label = chat.projectSlug === null ? GENERAL : projects.find((project) => project.slug === chat.projectSlug)?.name ?? chat.projectSlug
      group = { key: `project:${key}`, label, items: [] }
      groups.set(key, group)
    }
    group.items.push({ chat })
  }
  return [...groups.values()]
}

/** `text` with the first occurrence of `query` (case-insensitive) bold. */
function Snippet({ text, query }: { text: string; query: string }) {
  const at = query ? text.toLowerCase().indexOf(query.toLowerCase()) : -1
  if (at < 0) return <>{text}</>
  return <>{text.slice(0, at)}<strong className="font-semibold text-ink">{text.slice(at, at + query.length)}</strong>{text.slice(at + query.length)}</>
}

function ChatRow({ entry: { chat, match, first = true }, query, now }: { entry: Entry; query: string; now: number }) {
  if (match && !first) {
    return <span className={cn("truncate py-0.5 text-[10px] text-ink-subtle", ROW_TEXT_WIDTH)}><Snippet text={match.snippet} query={query} /></span>
  }
  return <span className="flex min-w-0 flex-1 flex-col py-0.5">
    <span className={cn("truncate leading-5", ROW_TEXT_WIDTH)}>{chat.title}</span>
    <span className={cn("truncate text-[10px] text-ink-subtle", !match && "tabular-nums", ROW_TEXT_WIDTH)}>
      {match ? <Snippet text={match.snippet} query={query} /> : compactElapsed(Date.parse(chat.updatedAt), now)}
    </span>
  </span>
}

/**
 * Panel over every chat of the login, opened by Alt+C in the active tab and docked under the top bar.
 *
 * - No query: the chats grouped by project, the group with the newest chat first, chats newest first
 * - A query: chats whose title holds every word, each followed by its own content hits, then every content hit of the other chats, one row each
 * - Picking a chat opens it in the active tab, picking a hit reveals its pair and paints its match (`onOpenChat`)
 */
export function ChatPicker({ onOpenChat }: { onOpenChat: OpenChat }) {
  const { open, close } = useAltKeyPicker("c")
  return open ? <ChatPickerPanel onOpenChat={onOpenChat} onClose={close} /> : null
}

/** The open picker: mounted per opening, so each opening reads the chats afresh and starts with an empty query. */
function ChatPickerPanel({ onOpenChat, onClose }: { onOpenChat: OpenChat; onClose: () => void }) {
  const { projects } = useProject()
  const now = useNowMs()
  const [query, setQuery] = useState("")
  const { chats, error: listError } = useAllChats()
  const search = useChatSearch(query)

  // Cut as `useChatSearch` cuts it before sending, so the reveal paints what was searched
  const trimmed = query.trim().slice(0, QUERY_MAX_LENGTH)
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean)
  let groups: PickerGroup<Entry>[]
  if (words.length > 0) {
    const named = (chats ?? []).filter((chat) => words.every((word) => chat.title.toLowerCase().includes(word))).sort(newestFirst)
    const hitsOf = new Map((search.hits ?? []).map((hit) => [hit.historyFile, hit]))
    // A chat whose title matches leads with its title row, its content hits follow as rows of their own
    const namedItems = named.flatMap((chat): Entry[] => [{ chat }, ...(hitsOf.get(chat.historyFile)?.matches ?? []).map((match) => ({ chat, match, first: false }))])
    const shown = new Set(named.map((chat) => chat.historyFile))
    groups = [
      { key: "named", label: "Chats", items: namedItems },
      { key: "content", label: "Content", items: (search.hits ?? []).filter((hit) => !shown.has(hit.historyFile)).flatMap((hit): Entry[] => {
        const chat: ChatListing = { historyFile: hit.historyFile, projectSlug: hit.projectSlug, title: hit.title, updatedAt: hit.updatedAt }
        return hit.matches.length > 0 ? hit.matches.map((match, index) => ({ chat, match, first: index === 0 })) : [{ chat }]
      }) },
    ]
  } else {
    groups = groupByProject(chats ?? [], projects)
  }
  const error = search.error || listError
  const loading = words.length > 0 ? search.hits === null : chats === null

  return <FloatingPicker<Entry>
    label="Chats"
    onClose={onClose}
    groups={groups}
    itemKey={entryKey}
    searchPlaceholder="Search chats…"
    query={query}
    onQueryChange={setQuery}
    header={error ? <ErrorText className="text-[10px]">{error}</ErrorText> : undefined}
    emptyText={loading ? (words.length > 0 ? "Searching…" : "Loading…") : "No chats"}
    renderItem={(entry) => <ChatRow entry={entry} query={trimmed} now={now} />}
    onPick={({ chat, match }) => {
      if (match) onOpenChat(chat.historyFile, match.qaIndex, { query: trimmed, role: match.role, ordinal: match.ordinal })
      else onOpenChat(chat.historyFile)
      return true
    }}
  />
}
