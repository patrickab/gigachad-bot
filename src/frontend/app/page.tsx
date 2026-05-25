"use client"

import { useCallback, useState } from "react"
import { Sidebar } from "@/components/Sidebar"
import { ChatContainer } from "@/components/ChatContainer"
import { ModelSelector } from "@/components/ModelSelector"
import { MoreOptionsMenu } from "@/components/MoreOptionsMenu"
import { ReasoningSelector } from "@/components/ReasoningSelector"
import { ResearchModelsBar } from "@/components/ResearchModelsBar"
import { ThemeToggle } from "@/components/ThemeToggle"
import { useChat } from "@/hooks/useChat"

export default function Home() {
  const {
    messages,
    isStreaming,
    send,
    cancel,
    research,
    webSearch,
    reset,
    models,
    prompts,
    histories,
    refreshHistories,
    loadHistory,
    deleteMessagePair,
  } = useChat()

  const [collapsed, setCollapsed] = useState(false)
  const [selectedModel, setSelectedModel] = useState("")
  const [selectedPrompt, setSelectedPrompt] = useState<string | null>(null)
  const [temperature, setTemperature] = useState(0.2)
  const [topP, setTopP] = useState(0.95)
  const [reasoningEffort, setReasoningEffort] = useState("none")

  const [researchEnabled, setResearchEnabled] = useState(false)
  const [researchFastModel, setResearchFastModel] = useState("")
  const [researchSmartModel, setResearchSmartModel] = useState("")
  const [researchStrategicModel, setResearchStrategicModel] = useState("")
  const [researchDepth, setResearchDepth] = useState(2)
  const [researchBreadth, setResearchBreadth] = useState(4)
  const [researchReasoning, setResearchReasoning] = useState("medium")
  const [researchReportType, setResearchReportType] = useState("deep")

  const [webSearchEnabled, setWebSearchEnabled] = useState(false)
  const [webSearchNumQueries, setWebSearchNumQueries] = useState(3)
  const [webSearchResultsPerQuery, setWebSearchResultsPerQuery] = useState(5)

  const toggleResearch = () => {
    setResearchEnabled(!researchEnabled)
    if (webSearchEnabled) setWebSearchEnabled(false)
  }

  const toggleWebSearch = () => {
    setWebSearchEnabled(!webSearchEnabled)
    if (researchEnabled) setResearchEnabled(false)
  }

  const handleSend = useCallback(
    (text: string, imageDataUrl: string | null) => {
      if (webSearchEnabled) {
        webSearch({
          query: text,
          numQueries: webSearchNumQueries,
          resultsPerQuery: webSearchResultsPerQuery,
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
        })
      }
    },
    [
      webSearchEnabled, webSearchNumQueries, webSearchResultsPerQuery,
      researchEnabled, researchFastModel, researchSmartModel, researchStrategicModel,
      researchDepth, researchBreadth, researchReasoning, researchReportType,
      selectedModel, selectedPrompt, temperature, topP, reasoningEffort,
      send, research, webSearch,
    ]
  )

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed(!collapsed)}
        histories={histories}
        onHistoryLoad={loadHistory}
        onHistoryRefresh={refreshHistories}
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
              researchEnabled={researchEnabled}
              researchDepth={researchDepth}
              onResearchDepthChange={setResearchDepth}
              researchBreadth={researchBreadth}
              onResearchBreadthChange={setResearchBreadth}
              researchReasoning={researchReasoning}
              onResearchReasoningChange={setResearchReasoning}
              researchReportType={researchReportType}
              onResearchReportTypeChange={setResearchReportType}
              webSearchEnabled={webSearchEnabled}
              webSearchNumQueries={webSearchNumQueries}
              onWebSearchNumQueriesChange={setWebSearchNumQueries}
              webSearchResultsPerQuery={webSearchResultsPerQuery}
              onWebSearchResultsPerQueryChange={setWebSearchResultsPerQuery}
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
            webSearchEnabled={webSearchEnabled}
            onWebSearchToggle={toggleWebSearch}
          />
        </div>
      </main>
    </div>
  )
}
