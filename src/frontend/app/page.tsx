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
import { useModeState } from "@/hooks/useModeState"

const MODE_LABELS: Record<string, string> = {
  research: "Deep Research",
  search: "Search",
  ocr: "LaTeX OCR",
  chat: "Chat",
}

const DEFAULT_VISION_MODEL = "ollama/qwen3-vl:235b-instruct-cloud"

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
  } = useChat()

  const {
    researchEnabled,
    morphicSearchEnabled,
    ocrEnabled,
    toggleResearch,
    toggleMorphicSearch,
    toggleOCR,
  } = useModeState()

  const [collapsed, setCollapsed] = useState(true)
  const [selectedModel, setSelectedModel] = useState("")
  const [selectedPrompt, setSelectedPrompt] = useState<string | null>(null)
  const [temperature, setTemperature] = useState(0.2)
  const [topP, setTopP] = useState(0.95)
  const [reasoningEffort, setReasoningEffort] = useState("none")

  const [researchFastModel, setResearchFastModel] = useState("")
  const [researchSmartModel, setResearchSmartModel] = useState("")
  const [researchStrategicModel, setResearchStrategicModel] = useState("")
  const [researchDepth, setResearchDepth] = useState(2)
  const [researchBreadth, setResearchBreadth] = useState(4)
  const [researchReasoning, setResearchReasoning] = useState("medium")
  const [researchReportType, setResearchReportType] = useState("deep")

  const [searchDepth, setSearchDepth] = useState<"quick" | "adaptive">("adaptive")

  const [ocrModel, setOCRModel] = useState(DEFAULT_VISION_MODEL)
  const [ocrImage, setOCRImage] = useState<string | null>(null)
  const [downscaleImages, setDownscaleImages] = useState(true)

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
    (text: string, imageDataUrl: string | null) => {
      if (morphicSearchEnabled) {
        morphicSearch({
          query: text,
          searchDepth,
          model: selectedModel || undefined,
        })
      } else if (researchEnabled) {
        research({
          query: text,
          fastModel: researchFastModel,
          smartModel: researchSmartModel,
          strategicModel: researchStrategicModel,
          depth: researchDepth,
          breadth: researchBreadth,
          reasoningEffort: researchReasoning,
          reportType: researchReportType,
        })
      } else {
        send({
          model: selectedModel,
          user_msg: text,
          system_prompt: selectedPrompt ?? "",
          temperature,
          top_p: topP,
          reasoning_effort: reasoningEffort === "none" ? null : reasoningEffort,
          img_base64: imageDataUrl,
          downscale_images: downscaleImages,
        })
      }
    },
    [
      morphicSearchEnabled, searchDepth,
      researchEnabled, researchFastModel, researchSmartModel, researchStrategicModel,
      researchDepth, researchBreadth, researchReasoning, researchReportType,
      selectedModel, selectedPrompt, temperature, topP, reasoningEffort, downscaleImages,
      send, research, morphicSearch,
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
              fastModel={researchFastModel}
              smartModel={researchSmartModel}
              strategicModel={researchStrategicModel}
              onFastModelChange={setResearchFastModel}
              onSmartModelChange={setResearchSmartModel}
              onStrategicModelChange={setResearchStrategicModel}
            />
          ) : (
            <div className="flex items-center gap-4">
              <ModelSelector
                models={models}
                selectedModel={selectedModel}
                onSelect={setSelectedModel}
              />
              <div className="w-px h-4 bg-zinc-800" />
              <ReasoningSelector reasoningEffort={reasoningEffort} onReasoningChange={setReasoningEffort} />
            </div>
          )}

          <div className="flex-1" />

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <MoreOptionsMenu
              prompts={prompts}
              selectedPrompt={selectedPrompt}
              onPromptSelect={setSelectedPrompt}
              temperature={temperature}
              onTemperatureChange={setTemperature}
              topP={topP}
              onTopPChange={setTopP}
              onRefresh={refreshHistories}
              onReset={reset}
              messages={messages}
              researchEnabled={researchEnabled}
              researchDepth={researchDepth}
              onResearchDepthChange={setResearchDepth}
              researchBreadth={researchBreadth}
              onResearchBreadthChange={setResearchBreadth}
              researchReasoning={researchReasoning}
              onResearchReasoningChange={setResearchReasoning}
              researchReportType={researchReportType}
              onResearchReportTypeChange={setResearchReportType}
              morphicSearchEnabled={morphicSearchEnabled}
              searchDepth={searchDepth}
              onSearchDepthChange={setSearchDepth}
              ocrEnabled={ocrEnabled}
              ocrModel={ocrModel}
              onOCRModelChange={setOCRModel}
              downscaleImages={downscaleImages}
              onDownscaleImagesChange={setDownscaleImages}
              models={models}
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
            researchEnabled={researchEnabled}
            onResearchToggle={toggleResearch}
            morphicSearchEnabled={morphicSearchEnabled}
            onMorphicSearchToggle={toggleMorphicSearch}
            ocrEnabled={ocrEnabled}
            onOCRToggle={toggleOCR}
            onOCRRequest={setOCRImage}
            downscaleImages={downscaleImages}
          />
          {ocrEnabled && ocrImage && (
            <OCRPanel
              image={ocrImage}
              model={ocrModel || DEFAULT_VISION_MODEL}
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
    <TabManager
      renderContent={(tab, onModeLabel) => <TabContent tab={tab} onModeLabel={onModeLabel} />}
    />
  )
}