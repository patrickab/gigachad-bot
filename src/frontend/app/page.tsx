"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Sidebar } from "@/components/Sidebar"
import { ChatContainer } from "@/components/ChatContainer"
import { ModelSelector } from "@/components/ModelSelector"
import { MoreOptionsMenu } from "@/components/MoreOptionsMenu"
import { ReasoningSelector } from "@/components/ReasoningSelector"
import { ResearchModelsBar } from "@/components/ResearchModelsBar"
import { ThemeToggle } from "@/components/ThemeToggle"
import { OCRPanel } from "@/components/OCRPanel"
import { SaveChatModal } from "@/components/SaveChatModal"
import { TabManager } from "@/components/TabManager"
import type { Tab } from "@/components/TabManager"
import { useChat } from "@/hooks/useChat"
import { useModeState, ModeProvider } from "@/hooks/useModeState"
import { useSettings, SettingsProvider } from "@/contexts/SettingsContext"
import { handleStudyPdf, updateLastMsg as updateLastAssistant } from "@/hooks/useStudyHandler"
import { saveChatHistory as apiSaveChatHistory, deleteChatUploads, parseFiles, deleteAttachment } from "@/lib/api"
import { DEFAULT_VISION_MODEL } from "@/lib/config"
import type { Attachment, Message } from "@/lib/types"

const MODE_LABELS: Record<string, string> = {
  research: "Deep Research",
  search: "Search",
  ocr: "LaTeX OCR",
  study: "PDF Study",
  chat: "Chat",
}

function buildLLMMessage(text: string, attachments: Attachment[]): { userMsg: string; hiddenContent: string; imgBase64: string | null } {
  let userMsg = text
  let imgBase64: string | null = null

  const textParts: string[] = []
  const imgAtts: Attachment[] = []

  for (const a of attachments) {
    if (a.mime.startsWith("image/")) {
      imgAtts.push(a)
    } else {
      const content = a.parsedMd ?? a.content
      if (content) textParts.push(`### ${a.name}\n\n${content}`)
    }
  }

  if (imgAtts.length > 0) {
    imgBase64 = "pending"
  }

  const hiddenContent = textParts.length > 0
    ? "**Attached files:**\n\n" + textParts.join("\n\n") + "\n\n## END"
    : ""

  return { userMsg, hiddenContent, imgBase64 }
}

function defaultSendParams(settings: ReturnType<typeof useSettings>, overrides: Record<string, unknown> = {}) {
  return {
    model: settings.selectedModel,
    system_prompt: settings.selectedPrompt ?? "",
    temperature: settings.temperature,
    top_p: settings.topP,
    reasoning_effort: settings.reasoningEffort === "none" ? null : settings.reasoningEffort,
    downscale_images: settings.downscaleImages,
    img_base64: null,
    ...overrides,
  }
}

function TabContent({ tab, onModeLabel, onChatSaved }: { tab: Tab; onModeLabel: (label: string) => void; onChatSaved: (tabId: string, chatFilename: string) => void }) {
  const {
    messages,
    isStreaming,
    send,
    cancel,
    research,
    morphicSearch,
    reset,
    models,
    prompts,
    histories,
    historiesLoading,
    refreshHistories,
    loadHistory,
    deleteMessagePair,
    addMessagePair,
    setMessages,
  } = useChat()

  const {
    researchEnabled,
    morphicSearchEnabled,
    ocrEnabled,
    studyEnabled,
    toggleResearch,
    toggleMorphicSearch,
    toggleOCR,
    toggleStudy,
  } = useModeState()

  const settings = useSettings()

  const [collapsed, setCollapsed] = useState(true)
  const [ocrImage, setOCRImage] = useState<string | null>(null)
  const [chatId, setChatId] = useState(() => tab.chatId)
  const [saveModalOpen, setSaveModalOpen] = useState(false)

  const handleHistoryLoad = useCallback(async (filename: string) => {
    const result = await loadHistory(filename)
    if (result.chat_id) {
      setChatId(result.chat_id)
      onChatSaved(tab.id, filename)
    }
  }, [loadHistory, tab.id, onChatSaved])

  useEffect(() => {
    if (researchEnabled) onModeLabel("Deep Research")
    else if (morphicSearchEnabled) onModeLabel("Search")
    else if (ocrEnabled) onModeLabel("LaTeX OCR")
    else if (studyEnabled) onModeLabel("PDF Study")
    else onModeLabel("Chat")
  }, [researchEnabled, morphicSearchEnabled, ocrEnabled, studyEnabled, onModeLabel])

  const openSaveModal = useCallback(() => {
    setSaveModalOpen(true)
  }, [])

  const handleSaveSubmit = useCallback(async (name: string) => {
    const filename = name + ".json"
    onChatSaved(tab.id, filename)
    await apiSaveChatHistory(filename, messages, chatId)
    refreshHistories()
    setSaveModalOpen(false)
  }, [messages, chatId, tab.id, refreshHistories, onChatSaved])

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "s") { e.preventDefault(); openSaveModal() }
      if (e.altKey && e.key === "r") { e.preventDefault(); reset() }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [openSaveModal, reset])

  const handleSend = useCallback(
    async (text: string, attachments: Attachment[]) => {
      if (morphicSearchEnabled || researchEnabled) {
        if (attachments.length > 0) return
      }

      // Study mode with PDF attachment
      if (studyEnabled && attachments.some(a => a.mime === "application/pdf")) {
        await handleStudyPdf(text, attachments, chatId, settings.selectedModel, setMessages)
        toggleStudy()
        return
      }

      // Attachments (non-study)
      if (attachments.length > 0) {
        setMessages(prev => [
          ...prev,
          { role: "user" as const, content: text, attachments },
          { role: "assistant" as const, content: "" },
        ])

        const pdfNames = attachments.filter(a => a.mime === "application/pdf").map(a => a.name)
        let enriched = attachments

        if (pdfNames.length > 0) {
          try {
            const parsed = await parseFiles(chatId, pdfNames)
            enriched = attachments.map(a => {
              if (a.mime !== "application/pdf") return a
              const p = parsed.find(r => r.name === a.name)
              return { ...a, parsedMd: p?.parsedMd ?? undefined }
            })
            if (!text.trim()) {
              updateLastAssistant(setMessages, m => ({ ...m, content: "Documents parsed - how can I help you?" }))
              return
            }
          } catch { }
        }

        const { hiddenContent } = buildLLMMessage(text, enriched)

        setMessages(prev => {
          const copy = [...prev]
          const idx = copy.length - 2
          if (idx >= 0 && copy[idx].role === "user") copy[idx] = { ...copy[idx], attachments: enriched, hiddenContent }
          return copy
        })

        const imgAtt = enriched.find(a => a.mime.startsWith("image/"))
        let imgBase64: string | null = null
        if (imgAtt) {
          try {
            const resp = await fetch(imgAtt.url)
            const blob = await resp.blob()
            imgBase64 = await new Promise<string>(resolve => { const r = new FileReader(); r.onload = () => resolve(r.result as string); r.readAsDataURL(blob) })
          } catch { }
        }

        const llmMsg = hiddenContent ? (text ? `${hiddenContent}\n\n${text}` : hiddenContent) : text
        send({ ...defaultSendParams(settings), user_msg: llmMsg, img_base64: imgBase64, downscale_images: settings.downscaleImages }, true)
        return
      }

      // Mode dispatch (no attachments)
      if (morphicSearchEnabled) {
        morphicSearch({ query: text, searchDepth: settings.searchDepth, model: settings.selectedModel || undefined })
      } else if (researchEnabled) {
        research({
          query: text, fastModel: settings.researchFastModel, smartModel: settings.researchSmartModel, strategicModel: settings.researchStrategicModel,
          depth: settings.researchDepth, breadth: settings.researchBreadth, reasoningEffort: settings.researchReasoning, reportType: settings.researchReportType,
        })
      } else {
        send({ ...defaultSendParams(settings), user_msg: text })
      }
    },
    [morphicSearchEnabled, researchEnabled, studyEnabled, chatId, settings, send, research, morphicSearch, setMessages, toggleStudy],
  )

  const handleRemoveAttachment = useCallback((messageIndex: number, attachmentName: string) => {
    setMessages(prev => {
      const copy = [...prev]
      if (copy[messageIndex] && copy[messageIndex].attachments) {
        const att = copy[messageIndex].attachments!.find(a => a.name === attachmentName)
        if (att) deleteAttachment(chatId, attachmentName).catch(() => { })
        copy[messageIndex] = { ...copy[messageIndex], attachments: copy[messageIndex].attachments!.filter(a => a.name !== attachmentName) }
      }
      return copy
    })
  }, [setMessages, chatId])

  const handleAttachmentContentChange = useCallback((messageIndex: number, attachmentName: string, newContent: string) => {
    setMessages(prev => {
      const copy = [...prev]
      const msg = copy[messageIndex]
      if (msg?.attachments) {
        copy[messageIndex] = {
          ...msg,
          attachments: msg.attachments.map(a => a.name === attachmentName ? { ...a, content: newContent } : a),
        }
      }
      return copy
    })
  }, [setMessages])

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed(!collapsed)}
        histories={histories}
        historiesLoading={historiesLoading}
        onHistoryLoad={handleHistoryLoad}
        onHistoryRefresh={refreshHistories}
        onSave={openSaveModal}
        onReset={reset}
      />
      <main className="flex-1 min-w-0 flex flex-col relative bg-zinc-950">
        <header className="h-[60px] shrink-0 flex items-center px-4 gap-4 bg-zinc-950 z-40 border-b border-zinc-800/50">
          {researchEnabled ? (
            <ResearchModelsBar
              models={models}
              fastModel={settings.researchFastModel}
              smartModel={settings.researchSmartModel}
              strategicModel={settings.researchStrategicModel}
              onFastModelChange={settings.setResearchFastModel}
              onSmartModelChange={settings.setResearchSmartModel}
              onStrategicModelChange={settings.setResearchStrategicModel}
            />
          ) : (
            <div className="flex items-center gap-4">
              <ModelSelector models={models} selectedModel={settings.selectedModel} onSelect={settings.setSelectedModel} />
              <div className="w-px h-4 bg-zinc-800" />
              <ReasoningSelector reasoningEffort={settings.reasoningEffort} onReasoningChange={settings.setReasoningEffort} />
            </div>
          )}

          <div className="flex-1" />

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <MoreOptionsMenu prompts={prompts} />
          </div>
        </header>
        <div className="flex-1 overflow-hidden relative">
          <ChatContainer
            chatId={chatId}
            messages={messages}
            isStreaming={isStreaming}
            onSend={handleSend}
            onCancel={cancel}
            onDeletePair={deleteMessagePair}
            onOCRRequest={setOCRImage}
            onRemoveAttachment={handleRemoveAttachment}
            onAttachmentContentChange={handleAttachmentContentChange}
          />
          {ocrEnabled && ocrImage && (
            <OCRPanel
              image={ocrImage}
              model={settings.ocrModel || DEFAULT_VISION_MODEL}
              onComplete={(output) => { addMessagePair("OCR Request", output); setOCRImage(null) }}
              onClose={() => { setOCRImage(null); toggleOCR() }}
            />
          )}
        </div>
      </main>
      <SaveChatModal open={saveModalOpen} onClose={() => setSaveModalOpen(false)} onSave={handleSaveSubmit} />
    </div>
  )
}

export default function Home() {
  const tabChatFilenames = useRef<Map<string, string | null>>(new Map())

  const handleChatSaved = useCallback((tabId: string, chatFilename: string) => {
    tabChatFilenames.current.set(tabId, chatFilename)
  }, [])

  const handleTabClose = useCallback((tab: Tab) => {
    const filename = tabChatFilenames.current.get(tab.id)
    if (!filename) deleteChatUploads(tab.chatId).catch(() => { })
    tabChatFilenames.current.delete(tab.id)
  }, [])

  return (
    <SettingsProvider>
      <TabManager
        onCloseTab={handleTabClose}
        renderContent={(tab, onModeLabel) => (
          <ModeProvider>
            <TabContent tab={tab} onModeLabel={onModeLabel} onChatSaved={handleChatSaved} />
          </ModeProvider>
        )}
      />
    </SettingsProvider>
  )
}
