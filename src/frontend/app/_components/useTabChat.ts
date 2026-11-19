import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { FolderOpen, Library } from "lucide-react"
import { ArchitectureIcon } from "@/components/ui/ArchitectureIcon"
import type { ChatFocus } from "@/features/chat/ChatContainer"
import type { AttachMenuItem, ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { effectiveReasoningEffort, useReasoningLevels } from "@/features/settings/ReasoningSelector"
import { useBranches } from "@/features/navigation/BranchContext"
import { itemOf, useTab, type Tab, type WorkspaceItem } from "@/contexts/TabContext"
import { useChat } from "@/features/chat/useChat"
import { useChatPersistence } from "@/features/chat/useChatPersistence"
import { useChatTurns } from "@/features/chat/useChatTurns"
import { useFilePicker } from "@/features/chat/useFilePicker"
import { useMemoryCommandBar } from "@/features/memory/useMemoryCommandBar"
import type { MemorizeSource } from "@/features/memory/useMemorizeFlow"
import { useProjectDocuments } from "@/features/chat/useProjectDocuments"
import { useSlashCommands } from "@/features/navigation/useSlashCommands"
import { piSessionHistory, type PiShellCommands } from "@/features/pi/PiTab"
import { fetchOmpSessionHistory, loadChatHistory } from "@/lib/backend/api"
import type { Message } from "@/lib/domain/types"
import { liveReferencePaths, normalizeMessageAttachments } from "@/features/chat/attachments"
import { parseHistoryFile } from "@/lib/domain/historyFile"
import { isEditableTarget, isModKey, isTerminalKeyEvent } from "@/lib/utils"
import { useBusyCount } from "@/hooks/useBusyCount"
import { useEventListener } from "@/hooks/useEventListener"

/** The open Pi session `sessionId`'s history, or why there is none. */
async function piSource(sessionId: string): Promise<MemorizeSource> {
  const messages = piSessionHistory(sessionId)
  if (!messages) throw new Error("The Pi session is not open yet")
  return { messages, conversationId: sessionId }
}

/** What `/memorize` reads in `tab`: the Pi or omp session an agent tab shows, else the chat, never a branch chat. Null when there is none. */
export function useMemorizeSource(tab: Tab, messages: Message[], chatId: string, branchMessageIdx: number | null): (() => Promise<MemorizeSource>) | null {
  const { appMode } = tab
  const ompView = appMode === "omp" ? itemOf(tab, "omp")?.omp : null
  const ompSessionId = ompView?.kind === "session" ? ompView.sessionId : null
  const piSessionId = appMode === "pi" ? itemOf(tab, "pi")?.pi?.sessionId ?? null : null
  return useMemo(() => {
    if (appMode === "pi") return piSessionId ? () => piSource(piSessionId) : null
    if (appMode === "omp") return ompSessionId ? async () => ({ messages: await fetchOmpSessionHistory(ompSessionId), conversationId: ompSessionId }) : null
    return branchMessageIdx == null ? async () => ({ messages, conversationId: chatId }) : null
  }, [appMode, piSessionId, ompSessionId, branchMessageIdx, messages, chatId])
}

/** Run one tab's chat (history, turns, attachments, memory commands, shortcuts); called in every mode so the chat survives mode switches. */
export function useTabChat({ focus, onOpenChat, onOpenItem }: { focus: ChatFocus; onOpenChat: (historyFile: string) => void; onOpenItem: (item: WorkspaceItem) => void }) {
  const { tab, isActive, updateConfig, setHistoryFile, setTitle } = useTab()
  const { projectSlug, config, appMode } = tab
  const chat = useChat()
  const { messages, cancel, reset, models, prompts, setMessages, totalUsage, setTotalUsage } = chat
  const { refreshAll, createBranch, setActiveFile, setActiveQaIndex } = useBranches()
  const reasoningLevels = useReasoningLevels(config.selectedModel)
  const reasoningEffort = effectiveReasoningEffort(reasoningLevels, config.reasoningEffort)
  const usage = totalUsage.total_tokens > 0 ? totalUsage : undefined

  const [chatId, setChatId] = useState(() => tab.chatId)
  const [loadedBranch, setLoadedBranch] = useState<{ historyFile: string | undefined; messageIdx: number | null }>({ historyFile: undefined, messageIdx: null })
  // Index where this chat branched off its parent, null for a main chat or while its history loads.
  const historyLoaded = loadedBranch.historyFile === tab.historyFile
  const branchMessageIdx = historyLoaded ? loadedBranch.messageIdx : null
  // A saved chat whose history has not arrived yet, so its empty messages do not mean an empty chat. A failed load stays here.
  const historyLoading = !!tab.historyFile && !historyLoaded
  const { busy: extracting, track } = useBusyCount()
  const chatInputRef = useRef<ChatInputHandle>(null)

  const filePicker = useFilePicker({ hasMessages: messages.length > 0, chatInputRef, track, onOpenItem })
  const filesEnabled = filePicker.rootFiles.enabled
  const docs = useProjectDocuments({ isActive, appMode, projectSlug, chatId, chatInputRef, track })
  const persistence = useChatPersistence({ tab, projectSlug, messages, chatId, usage, refreshAll, setHistoryFile })
  const memory = useMemoryCommandBar({ loadSource: useMemorizeSource(tab, messages, chatId, branchMessageIdx) })
  const { menuItems, runSlashCommand } = useSlashCommands({
    filesEnabled: filesEnabled && appMode === "chat",
    openFilePicker: filePicker.openFilePicker,
    memorize: memory.memorize,
  })
  const shellCommands = useMemo<PiShellCommands>(() => ({ names: menuItems.map((item) => item.command.slice(1)), run: runSlashCommand }), [menuItems, runSlashCommand])
  const turns = useChatTurns({
    chat,
    chatId,
    projectSlug,
    config,
    reasoningEffort,
    prompts,
    autosave: persistence.handleAutosave,
    runSlashCommand,
  })
  const { syncReferences } = turns

  // A new tab has a blank selectedModel (DEFAULT_TAB_CONFIG). Seed it from the
  // configured default chat model. Never hardcode a fallback name here.
  useEffect(() => {
    if (!config.selectedModel && models?.defaults.default_model) updateConfig({ selectedModel: models.defaults.default_model })
  }, [config.selectedModel, models, updateConfig])

  useEffect(() => {
    if (isActive) setActiveFile(tab.historyFile)
  }, [isActive, tab.historyFile, setActiveFile])

  useEffect(() => {
    if (isActive) setActiveQaIndex(focus.qaIndex)
  }, [isActive, focus.qaIndex, setActiveQaIndex])
  useEffect(() => cancel, [cancel])

  useEffect(() => {
    if (!tab.historyFile) return
    const historyFile = tab.historyFile
    let stale = false
    setMessages([])
    setTotalUsage({ total_tokens: 0, prompt_tokens: 0, completion_tokens: 0 })
    // Project chats load by their own slug, so switching projects never reloads open tabs.
    const { slug } = parseHistoryFile(historyFile)
    loadChatHistory(historyFile).then((data) => {
      if (stale) return
      if (data.messages.length > 0) {
        const loaded = data.messages.map((m) => normalizeMessageAttachments(m, data.chat_id, slug))
        setMessages(loaded)
        void syncReferences(liveReferencePaths(loaded.flatMap((m) => m.attachments ?? [])))
      }
      if (data.chat_id) setChatId(data.chat_id)
      if (data.title) setTitle(data.title)
      if (data.usage) setTotalUsage(data.usage)
      setLoadedBranch({ historyFile, messageIdx: data.branch_message_idx ?? null })
    }).catch(() => { })
    return () => { stale = true }
  }, [tab.historyFile, setMessages, setTitle, setTotalUsage, syncReferences])

  // Capture phase so the shortcuts win over handlers inside focused editors. A Pi tab keeps only Alt+M (memorize its session), the rest belong to the chat.
  useEventListener("window", "keydown", (e) => {
    if (isTerminalKeyEvent(e)) return
    if (e.altKey && e.key.toLowerCase() === "m") {
      e.preventDefault()
      if (memory.isIdle) void memory.memorize()
    }
    if (appMode !== "chat") return
    if (e.altKey && e.key === "s") {
      e.preventDefault()
      if (persistence.isTitled) persistence.handleQuickSave()
      else persistence.setSaveModalOpen(true)
    }
    if (e.altKey && e.key === "r") { e.preventDefault(); reset() }
    if (e.altKey && e.key.toLowerCase() === "o") {
      e.preventDefault()
      if (filesEnabled) filePicker.openFilePicker()
    }

    const isSlash = e.key === "/" && !e.altKey && !e.ctrlKey && !e.metaKey
    const isCtrlK = isModKey(e) && e.key.toLowerCase() === "k"

    if (isSlash || isCtrlK) {
      if (isSlash && isEditableTarget(e.target)) return
      e.preventDefault()
      e.stopPropagation()
      e.stopImmediatePropagation()
      if (memory.isIdle) {
        memory.openMenu()
      }
    }
  }, { active: isActive && (appMode === "chat" || appMode === "pi"), capture: true })

  // Picking a menu entry always closes the menu first; memorizing then replaces it with the review panel.
  const { closeMenu: closeMemoryMenu } = memory
  const runMenuCommand = useCallback((command: string) => {
    closeMemoryMenu()
    void runSlashCommand(command)
  }, [closeMemoryMenu, runSlashCommand])

  const attachmentActions = useMemo(() => ({
    onToggleActive: turns.handleToggleAttachmentActive,
    onRemove: turns.handleRemoveAttachment,
    onContentChange: turns.handleAttachmentContentChange,
    onPreviewOpen: (path: string) => syncReferences([path]),
  }), [turns.handleToggleAttachmentActive, turns.handleRemoveAttachment, turns.handleAttachmentContentChange, syncReferences])

  const { handleAttachArchitecture } = docs
  const attachMenuItems = useMemo<AttachMenuItem[]>(() => [
    ...(filesEnabled ? [{ label: "Mounted file", icon: FolderOpen, onClick: filePicker.openFilePicker }] : []),
    ...(projectSlug ? [{ label: "Project document", icon: Library, onClick: docs.openDocuments }] : []),
    ...(handleAttachArchitecture ? [{ label: "Architecture", icon: ArchitectureIcon, onClick: () => void handleAttachArchitecture() }] : []),
  ], [filesEnabled, filePicker.openFilePicker, projectSlug, docs.openDocuments, handleAttachArchitecture])

  const handleBranch = useCallback((qaIndex: number) => {
    if (!tab.historyFile) return
    createBranch(tab.historyFile, qaIndex).then(({ childFile }) => {
      onOpenChat(childFile)
    })
  }, [tab.historyFile, createBranch, onOpenChat])

  return {
    chat, chatId, branchMessageIdx, historyLoading, extracting, chatInputRef, reasoningLevels, reasoningEffort,
    filePicker, docs, persistence, memory, menuItems, runMenuCommand, shellCommands, turns, attachmentActions, attachMenuItems,
    onBranch: tab.historyFile ? handleBranch : undefined,
  }
}
