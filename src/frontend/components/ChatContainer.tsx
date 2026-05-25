"use client"

import { useEffect, useRef } from "react"
import { AnimatePresence } from "framer-motion"
import type { Message } from "@/lib/types"
import { ChatMessage } from "./ChatMessage"
import { ChatInput } from "./ChatInput"
import { cn } from "@/lib/utils"

interface ChatContainerProps {
  messages: Message[]
  isStreaming: boolean
  onSend: (text: string, imageDataUrl: string | null) => void
  onCancel: () => void
  onDeletePair: (index: number) => void
  className?: string
  researchEnabled?: boolean
  onResearchToggle?: () => void
  webSearchEnabled?: boolean
  onWebSearchToggle?: () => void
  ocrEnabled?: boolean
  onOCRToggle?: () => void
  onOCRRequest?: (image: string) => void
  downscaleImages?: boolean
}

export function ChatContainer({
  messages,
  isStreaming,
  onSend,
  onCancel,
  onDeletePair,
  className,
  researchEnabled,
  onResearchToggle,
  webSearchEnabled,
  onWebSearchToggle,
  ocrEnabled,
  onOCRToggle,
  onOCRRequest,
  downscaleImages,
}: ChatContainerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

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
    <div className={cn("flex flex-col h-full relative", className)}>
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
              <ChatMessage role="assistant" content={assistant.content} index={globalIndex} onDelete={onDeletePair} />
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
          researchEnabled={researchEnabled}
          onResearchToggle={onResearchToggle}
          webSearchEnabled={webSearchEnabled}
          onWebSearchToggle={onWebSearchToggle}
          ocrEnabled={ocrEnabled}
          onOCRToggle={onOCRToggle}
          downscaleImages={downscaleImages}
        />
      </div>
    </div>
  )
}
