"use client"

import { useCallback, useEffect, useState } from "react"
import { Sidebar } from "@/components/Sidebar"
import { ChatContainer } from "@/components/ChatContainer"
import { ModelSelector } from "@/components/ModelSelector"
import { MoreOptionsMenu } from "@/components/MoreOptionsMenu"
import { ReasoningSelector } from "@/components/ReasoningSelector"
import { ResearchModelsBar } from "@/components/ResearchModelsBar"
import { ThemeToggle } from "@/components/ThemeToggle"
import { OCRPanel } from "@/components/OCRPanel"
import { TabManager } from "@/components/TabManager"
import type { Tab } from "@/components/TabManager"
import { useChat } from "@/hooks/useChat"
import { useModeState, ModeProvider } from "@/hooks/useModeState"
import { useSettings, SettingsProvider } from "@/contexts/SettingsContext"
import { parsePdfs as apiParsePdfs } from "@/lib/api"
import { DEFAULT_VISION_MODEL } from "@/lib/config"

const MODE_LABELS: Record<string, string> = {
  research: "Deep Research",
  search: "Search",
  ocr: "LaTeX OCR",
  chat: "Chat",
}

function TabContent({ tab, onModeLabel }: { tab: Tab; onModeLabel: (label: string) => void }) {
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
    toggleResearch,
    toggleMorphicSearch,
    toggleOCR,
  } = useModeState()

  const settings = useSettings()

  const [collapsed, setCollapsed] = useState(true)
  const [ocrImage, setOCRImage] = useState<string | null>(null)

  useEffect(() => {
    if (researchEnabled) onModeLabel("Deep Research")
    else if (morphicSearchEnabled) onModeLabel("Search")
    else if (ocrEnabled) onModeLabel("LaTeX OCR")
    else onModeLabel("Chat")
  }, [researchEnabled, morphicSearchEnabled, ocrEnabled, onModeLabel])

  const handleSidebarSave = useCallback(async () => {
    const name = window.prompt("Enter name for chat:")
    if (name?.trim()) {
      const { saveChatHistory } = await import("@/lib/api")
      await saveChatHistory(`${name.trim()}.json`, messages)
      refreshHistories()
    }
  }, [messages, refreshHistories])

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "s") {
        e.preventDefault()
        handleSidebarSave()
      }
      if (e.altKey && e.key === "r") {
        e.preventDefault()
        reset()
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [handleSidebarSave, reset])

  const handleSend = useCallback(
    async (text: string, imageDataUrl: string | null, pdfFiles?: File[] | null) => {
      if (pdfFiles && pdfFiles.length > 0) {
        if (morphicSearchEnabled || researchEnabled) {
          return
        }
        const query = text.trim()
        const label = pdfFiles.length === 1 ? pdfFiles[0].name : `${pdfFiles.length} PDFs`
        addMessagePair("📄 " + label, "Processing...")
        try {
          const batch = await apiParsePdfs(pdfFiles, query, "pipeline", settings.selectedModel)
          const merged = batch.results
            .map((r) => {
              const content = r.answer ?? r.markdown_content
              return `### ${r.filename}\n\n${content}`
            })
            .join("\n\n---\n\n")
          setMessages((prev) => {
            const copy = [...prev]
            copy[copy.length - 1] = { role: "assistant", content: merged }
            return copy
          })
        } catch {
          setMessages((prev) => {
            const copy = [...prev]
            copy[copy.length - 1] = { role: "assistant", content: "Error: Failed to parse PDFs." }
            return copy
          })
        }
        return
      }

      if (morphicSearchEnabled) {
        morphicSearch({
          query: text,
          searchDepth: settings.searchDepth,
          model: settings.selectedModel || undefined,
        })
      } else if (researchEnabled) {
        research({
          query: text,
          fastModel: settings.researchFastModel,
          smartModel: settings.researchSmartModel,
          strategicModel: settings.researchStrategicModel,
          depth: settings.researchDepth,
          breadth: settings.researchBreadth,
          reasoningEffort: settings.researchReasoning,
          reportType: settings.researchReportType,
        })
      } else {
        send({
          model: settings.selectedModel,
          user_msg: text,
          system_prompt: settings.selectedPrompt ?? "",
          temperature: settings.temperature,
          top_p: settings.topP,
          reasoning_effort: settings.reasoningEffort === "none" ? null : settings.reasoningEffort,
          img_base64: imageDataUrl,
          downscale_images: settings.downscaleImages,
        })
      }
    },
    [
      morphicSearchEnabled, settings.searchDepth,
      researchEnabled, settings.researchFastModel, settings.researchSmartModel, settings.researchStrategicModel,
      settings.researchDepth, settings.researchBreadth, settings.researchReasoning, settings.researchReportType,
      settings.selectedModel, settings.selectedPrompt, settings.temperature, settings.topP, settings.reasoningEffort, settings.downscaleImages,
      send, research, morphicSearch, addMessagePair, setMessages,
    ]
  )

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed(!collapsed)}
        histories={histories}
        historiesLoading={historiesLoading}
        onHistoryLoad={loadHistory}
        onHistoryRefresh={refreshHistories}
        onSave={handleSidebarSave}
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
              <ModelSelector
                models={models}
                selectedModel={settings.selectedModel}
                onSelect={settings.setSelectedModel}
              />
              <div className="w-px h-4 bg-zinc-800" />
              <ReasoningSelector reasoningEffort={settings.reasoningEffort} onReasoningChange={settings.setReasoningEffort} />
            </div>
          )}

          <div className="flex-1" />

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <MoreOptionsMenu
              prompts={prompts}
            />
          </div>
        </header>
        <div className="flex-1 overflow-hidden relative">
          <ChatContainer
            messages={messages}
            isStreaming={isStreaming}
            onSend={handleSend}
            onCancel={cancel}
            onDeletePair={deleteMessagePair}
            onOCRRequest={setOCRImage}
          />
          {ocrEnabled && ocrImage && (
            <OCRPanel
              image={ocrImage}
              model={settings.ocrModel || DEFAULT_VISION_MODEL}
              onComplete={(output) => {
                addMessagePair("OCR Request", output)
                setOCRImage(null)
              }}
              onClose={() => {
                setOCRImage(null)
                toggleOCR()
              }}
            />
          )}
        </div>
      </main>
    </div>
  )
}

export default function Home() {
  return (
    <SettingsProvider>
      <ModeProvider>
        <TabManager
          renderContent={(tab, onModeLabel) => <TabContent tab={tab} onModeLabel={onModeLabel} />}
        />
      </ModeProvider>
    </SettingsProvider>
  )
}