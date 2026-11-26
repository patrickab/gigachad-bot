"use client"

// How a factory event (a `factory-notice`, a `factory` tool result) shows in a Pi tab: its card (contract `cards`),
// or the agent's text instead. Readable is the default; Alt+V switches the whole tab (remembered per tab in
// sessionStorage), a card's own switch flips just that card.

import { createContext, useContext, useState, useSyncExternalStore } from "react"
import { CornerUpLeft, Factory, FileText, LayoutList } from "lucide-react"
import { Disclosure } from "@/components/ui/Disclosure"
import { IconButton } from "@/components/ui/IconButton"
import { useEventListener } from "@/hooks/useEventListener"
import { isAltOnly, isTerminalKeyEvent } from "@/lib/utils"
import { FactoryCard } from "./FactoryCard"
import type { FactoryCard as Card } from "./factoryCard"

/** sessionStorage key prefix of a tab's switch, the tab id follows. */
export const FACTORY_VIEW_STORAGE_KEY = "gc.pi.factoryView"

export interface FactoryView {
  /** The tab shows the agent's text instead of the cards. */
  agentText: boolean
  /** Opens a finding's file, at its line. */
  openFile?: (path: string, line?: number) => void
  /** Scrolls to the notice card keyed `key`, what a decision answers. */
  revealNotice?: (key: string) => void
  /** The keys of the transcript's notices: a decision links only to one of them. */
  notices?: NoticeKeys
}

/** The keys of a transcript's notices, kept in one object so its events re-render only when theirs comes or goes. */
export interface NoticeKeys {
  has: (key: string) => boolean
  subscribe: (onChange: () => void) => () => void
  set: (keys: ReadonlySet<string>) => void
}

export function createNoticeKeys(): NoticeKeys {
  let keys: ReadonlySet<string> = new Set()
  const listeners = new Set<() => void>()
  return {
    has: (key) => keys.has(key),
    subscribe: (onChange) => {
      listeners.add(onChange)
      return () => { listeners.delete(onChange) }
    },
    set: (next) => {
      keys = next
      for (const listener of listeners) listener()
    },
  }
}

const NO_NOTICES: NoticeKeys["subscribe"] = () => () => {}

const FactoryViewContext = createContext<FactoryView>({ agentText: false })

/** Provided by the transcript: keep its value stable, every factory event of the tab reads it. */
export const FactoryViewProvider = FactoryViewContext.Provider

function stored(tabId: string | null): boolean {
  if (tabId === null) return false
  try {
    return sessionStorage.getItem(`${FACTORY_VIEW_STORAGE_KEY}:${tabId}`) === "text"
  } catch {
    return false
  }
}

function store(tabId: string | null, agentText: boolean) {
  if (tabId === null) return
  try {
    if (agentText) sessionStorage.setItem(`${FACTORY_VIEW_STORAGE_KEY}:${tabId}`, "text")
    else sessionStorage.removeItem(`${FACTORY_VIEW_STORAGE_KEY}:${tabId}`)
  } catch {
    // Storage unavailable: the switch holds for this page only
  }
}

/** The tab's switch: Alt+V toggles it while `active`, from anywhere in the tab but a terminal. */
export function useTabAgentText(tabId: string | null, active: boolean): boolean {
  const [state, setState] = useState(() => ({ tabId, agentText: stored(tabId) }))
  // Another tab id (a remount would do the same) reads its own remembered switch
  const current = state.tabId === tabId ? state : { tabId, agentText: stored(tabId) }
  if (current !== state) setState(current)
  useEventListener("window", "keydown", (event) => {
    if (!isAltOnly(event) || (event.code !== "KeyV" && event.key.toLowerCase() !== "v") || isTerminalKeyEvent(event)) return
    event.preventDefault()
    const agentText = !current.agentText
    store(tabId, agentText)
    setState({ tabId, agentText })
  }, { active })
  return current.agentText
}

interface FactoryEventProps {
  card: Card
  /** The agent's text: the notice's content, the tool result's text. */
  text: string
  /** A notice's key, what a decision's `answers` names. */
  noticeKey?: string
}

/** One factory event: its card, or the agent's text; a switch on its row flips it, a decision links to what it answers. */
export function FactoryEvent({ card, text, noticeKey }: FactoryEventProps) {
  const { agentText: tabAgentText, openFile, revealNotice, notices } = useContext(FactoryViewContext)
  const [flipped, setFlipped] = useState(false)
  const agentText = tabAgentText !== flipped
  const answers = card.answers
  // Offered only while the answered notice is in this transcript
  const answered = useSyncExternalStore(notices?.subscribe ?? NO_NOTICES, () => !!answers && !!notices?.has(answers), () => false)
  const actions = (
    <>
      {answers && answered && revealNotice && <IconButton icon={CornerUpLeft} label="Show what this answers" onClick={() => revealNotice(answers)} />}
      <IconButton icon={agentText ? LayoutList : FileText} label={agentText ? "Show the card" : "Show the agent's text"} onClick={() => setFlipped((value) => !value)} />
    </>
  )
  return (
    <div className="my-1" data-factory-notice={noticeKey}>
      {agentText ? (
        <>
          <Disclosure icon={Factory} summary={<span className="min-w-0 flex-1 truncate">{card.title}</span>} actions={actions} />
          <pre data-agent-text className="m-0 whitespace-pre-wrap break-words pl-5 font-sans text-xs text-ink-muted">{text}</pre>
        </>
      ) : (
        <FactoryCard card={card} onOpenFile={openFile} actions={actions} className="" />
      )}
    </div>
  )
}
