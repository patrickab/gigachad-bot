import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { FolderOpen, Library } from "lucide-react"
import { ArchitectureIcon } from "@/components/ui/ArchitectureIcon"
import type { AttachMenuItem, ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { effectiveReasoningEffort, useReasoningLevels } from "@/features/settings/ReasoningSelector"
import { useBranchActions } from "@/features/navigation/BranchContext"
import { shownItem, useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { createChatStore } from "@/features/chat/chatStore"
import { useChatConfig } from "@/features/chat/useChatConfig"
import { useChatPersistence } from "@/features/chat/useChatPersistence"
import { useChatTurns } from "@/features/chat/useChatTurns"
import { useFilePicker } from "@/features/chat/useFilePicker"
import { useProjectDocuments } from "@/features/chat/useProjectDocuments"
import { loadChatHistory } from "@/lib/backend/api"
import { liveReferencePaths, normalizeMessageAttachments } from "@/features/chat/attachments"
import { parseHistoryFile } from "@/lib/domain/historyFile"
import { isTerminalKeyEvent } from "@/lib/utils"
import { useBusyCount } from "@/hooks/useBusyCount"
import { useStore } from "@/hooks/useStore"
import { useEventListener } from "@/hooks/useEventListener"

const NO_USAGE = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }

/** Run one tab's chat (history, turns, attachments, shortcuts); called in every mode so the chat survives mode switches. */
export function useTabChat({ onOpenChat, onOpenItem, onSaveAs }: {
  onOpenChat: (historyFile: string) => void
  onOpenItem: (item: WorkspaceItem) => void
  /** Asks for a name, then `save` saves the chat under it: the messages of that moment, adopted by this tab. */
  onSaveAs: (save: (name: string) => Promise<void>) => void
}) {
  const { tab, isActive, updateConfig, setHistoryFile, setTitle } = useTab()
  const { projectSlug, config } = tab
  const chatShown = shownItem(tab) === null
  const [chat] = useState(createChatStore)
  const hasMessages = useStore(chat, (state) => state.messages.length > 0)
  const isStreaming = useStore(chat, (state) => state.isStreaming)
  const chatConfig = useChatConfig()
  const { models, prompts } = chatConfig
  const { refreshAll, createBranch } = useBranchActions()
  const reasoningLevels = useReasoningLevels(config.selectedModel)
  const reasoningEffort = effectiveReasoningEffort(reasoningLevels, config.reasoningEffort)

  const [chatId, setChatId] = useState(() => tab.chatId)
  const [loadedBranch, setLoadedBranch] = useState<{ historyFile: string | undefined; messageIdx: number | null }>({ historyFile: undefined, messageIdx: null })
  // Index where this chat branched off its parent, null for a main chat or while its history loads.
  const historyLoaded = loadedBranch.historyFile === tab.historyFile
  const branchMessageIdx = historyLoaded ? loadedBranch.messageIdx : null
  // A saved chat whose history has not arrived yet, so its empty messages do not mean an empty chat. A failed load stays here.
  const historyLoading = !!tab.historyFile && !historyLoaded
  const { busy: extracting, track } = useBusyCount()
  const chatInputRef = useRef<ChatInputHandle>(null)

  const filePicker = useFilePicker({ hasMessages, chatInputRef, track, onOpenItem })
  const filesEnabled = filePicker.rootFiles.enabled
  const docs = useProjectDocuments({ isActive, chatShown, projectSlug, chatId, chatInputRef, track })
  const persistence = useChatPersistence({ tab, projectSlug, chat, chatId, refreshAll, setHistoryFile })
  const turns = useChatTurns({
    chat,
    chatId,
    projectSlug,
    config,
    reasoningEffort,
    prompts,
    save: persistence.handleSave,
    autosave: persistence.handleAutosave,
  })
  const { syncReferences } = turns

  // A new tab has a blank selectedModel (DEFAULT_TAB_CONFIG). Seed it from the
  // configured default chat model. Never hardcode a fallback name here.
  useEffect(() => {
    if (!config.selectedModel && models?.defaults.default_model) updateConfig({ selectedModel: models.defaults.default_model })
  }, [config.selectedModel, models, updateConfig])

  useEffect(() => chat.cancel, [chat])

  useEffect(() => {
    if (!tab.historyFile) return
    const historyFile = tab.historyFile
    let stale = false
    chat.update(() => [])
    chat.setUsage(NO_USAGE)
    // Project chats load by their own slug, so switching projects never reloads open tabs.
    const { slug } = parseHistoryFile(historyFile)
    loadChatHistory(historyFile).then((data) => {
      if (stale) return
      if (data.messages.length > 0) {
        const loaded = data.messages.map((m) => normalizeMessageAttachments(m, data.chat_id, slug))
        chat.update(() => loaded)
        void syncReferences(liveReferencePaths(loaded.flatMap((m) => m.attachments ?? [])))
      }
      if (data.chat_id) setChatId(data.chat_id)
      if (data.title) setTitle(data.title)
      if (data.usage) chat.setUsage(data.usage)
      setLoadedBranch({ historyFile, messageIdx: data.branch_message_idx ?? null })
    }).catch(() => { })
    return () => { stale = true }
  }, [tab.historyFile, chat, setTitle, syncReferences])

  // Capture phase so the shortcuts win over handlers inside focused editors.
  useEventListener("window", "keydown", (e) => {
    if (isTerminalKeyEvent(e)) return
    if (e.altKey && e.key === "s") {
      e.preventDefault()
      if (persistence.isTitled) persistence.handleQuickSave()
      else onSaveAs(persistence.handleSaveSubmit)
    }
    if (e.altKey && e.key === "r") {
      e.preventDefault()
      chat.update(() => [])
      chat.setUsage(NO_USAGE)
    }
    if (e.altKey && e.key.toLowerCase() === "o") {
      e.preventDefault()
      if (filesEnabled) filePicker.openFilePicker()
    }
  }, { active: isActive && chatShown, capture: true })

  // One object per chat: its members read the latest messages from the store, so the settled pairs they reach skip streamed frames
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

  // A send that failed after the cut keeps the edited text: it goes into the composer, ahead of any draft there. A cut that
  // failed leaves the pair, and `turns.turnError` says why
  const { handleEditPair } = turns
  const onEditPair = useCallback((index: number, text: string) => {
    void handleEditPair(index, text).then((outcome) => {
      if (outcome === "unsent") chatInputRef.current?.updateText((draft) => (draft ? `${text}\n\n${draft}` : text))
    })
  }, [handleEditPair])

  return {
    chat, hasMessages, isStreaming, chatConfig, chatId, branchMessageIdx, historyLoading, extracting, chatInputRef, reasoningLevels, reasoningEffort,
    filePicker, docs, turns, attachmentActions, attachMenuItems, onEditPair,
    onBranch: tab.historyFile ? handleBranch : undefined,
  }
}
