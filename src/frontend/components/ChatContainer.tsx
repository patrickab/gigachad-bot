"use client"

import { useEffect, useRef, useMemo } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message } from "@/lib/types"
import { ChatMessage } from "./ChatMessage"
import { ChatInput } from "./ChatInput"
import { SourcesSidebar } from "./SourcesSidebar"
import { cn } from "@/lib/utils"
import { useModeState } from "@/hooks/useModeState"
import { useSettings } from "@/contexts/SettingsContext"

interface ChatContainerProps {
  messages: Message[]
  isStreaming: boolean
  onSend: (text: string, imageDataUrl: string | null) => void
  onCancel: () => void
  onDeletePair: (index: number) => void
  className?: string
  onOCRRequest?: (image: string) => void
}

export function ChatContainer({
  messages,
  isStreaming,
  onSend,
  onCancel,
  onDeletePair,
  className,
  onOCRRequest,
}: ChatContainerProps) {
  const { researchEnabled, morphicSearchEnabled, ocrEnabled, toggleResearch, toggleMorphicSearch, toggleOCR } = useModeState()
  const { downscaleImages } = useSettings()
  const bottomRef = useRef<HTMLDivElement>(null)

  const lastMorphicResult = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].morphic_result) return messages[i].morphic_result
    }
    return undefined
  }, [messages])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const pairs: { user: Message; assistant: Message; globalIndex: number }[] = []
  for (let i = 0; i < messages.length; i += 2) {
    if (messages[i]?.role === "user") {
      pairs.push({
        user: messages[i],
        assistant: messages[i + 1] ?? { role: "assistant", content: "" },
        globalIndex: i,
      })
    }
  }

  return (
    <div className={cn("flex h-full relative", className)}>
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex-1 overflow-y-auto pb-6">
          {pairs.length === 0 && (
            <div className="flex h-full items-center justify-center">
              <p className="text-sm text-zinc-600">Send a message to start.</p>
            </div>
          )}
          <AnimatePresence>
            {pairs.map(({ user, assistant, globalIndex }) => (
              <div key={globalIndex} className="group">
                <ChatMessage role="user" content={user.content} index={globalIndex} onDelete={onDeletePair} />
                <ChatMessage role="assistant" content={assistant.content} morphic_result={assistant.morphic_result} index={globalIndex} onDelete={onDeletePair} />
              </div>
            ))}
          </AnimatePresence>
          <div ref={bottomRef} />
        </div>
        <div className="shrink-0 px-4 pb-6 z-10">
          <ChatInput
            onSend={onSend}
            onOCRRequest={onOCRRequest}
            disabled={isStreaming}
            isStreaming={isStreaming}
            onCancel={onCancel}
          />
        </div>
      </div>
      <SourcesSidebar result={lastMorphicResult} />
    </div>
  )
}