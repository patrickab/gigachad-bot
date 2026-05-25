"use client"

import { useCallback, useState } from "react"
import { Sidebar } from "@/components/Sidebar"
import { ChatContainer } from "@/components/ChatContainer"
import { ModelSelector } from "@/components/ModelSelector"
import { MoreOptionsMenu } from "@/components/MoreOptionsMenu"
import { ReasoningSelector } from "@/components/ReasoningSelector"
import { useChat } from "@/hooks/useChat"

export default function Home() {
  const {
    messages,
    isStreaming,
    send,
    cancel,
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

  const handleSend = useCallback(
    (text: string, imageDataUrl: string | null) => {
      send({
        model: selectedModel,
        user_msg: text,
        system_prompt: selectedPrompt ?? "",
        temperature,
        top_p: topP,
        reasoning_effort: reasoningEffort === "none" ? null : reasoningEffort,
        img_base64: imageDataUrl,
      })
    },
    [selectedModel, selectedPrompt, temperature, topP, reasoningEffort, send]
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
          <div className="flex items-center gap-4">
            <ModelSelector
              models={models}
              selectedModel={selectedModel}
              onSelect={setSelectedModel}
            />
            <div className="w-px h-4 bg-zinc-800" />
            <ReasoningSelector reasoningEffort={reasoningEffort} onReasoningChange={setReasoningEffort} />
          </div>

          <div className="flex-1" />

          <div className="flex items-center gap-4">
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
          />
        </div>
      </main>
    </div>
  )
}
