"use client"

import { useState, type ComponentProps } from "react"
import { withParsedAttachments } from "@/features/chat/attachments"
import { ChatInput } from "@/features/chat/composer/ChatInput"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import { parseFiles } from "@/lib/backend/api"
import type { Attachment, OmpView } from "@/lib/domain/types"
import { composerFor, ompModelOf, ompPrompt, ompSelectorModels, selectorModelId } from "./composer"
import { useOmp } from "./OmpProvider"

/** The frame's composer state when it shows `view`, null while the frame loads or shows another view. */
function useComposerFor(view: OmpView) {
  const { composer, frameStatus, sendToComposer } = useOmp()
  return { state: frameStatus === "ready" ? composerFor(view, composer) : null, sendToComposer }
}

/**
 * gigachad's composer for an omp tab. omp-web's own one stays hidden in the frame and submits what this one sends.
 *
 * - Images go as omp images, documents as text the way gigachad chats send them
 * - Disabled until the frame reports the composer of this tab's session
 */
export function OmpComposer({ view, chatId, slug }: { view: OmpView; chatId: string; slug: string | null }) {
  const { state, sendToComposer } = useComposerFor(view)
  const [preparing, setPreparing] = useState(false)
  const [error, setError] = useState("")
  const sessionId = state?.sessionId ?? null

  const send = async (text: string, attachments: Attachment[]) => {
    setError("")
    setPreparing(true)
    try {
      const prompt = await ompPrompt(text, attachments, (documents) => withParsedAttachments(documents, chatId, slug, parseFiles))
      sendToComposer({ type: "composer-send", sessionId, ...prompt })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not attach these files")
    } finally {
      setPreparing(false)
    }
  }

  return (
    <div className="px-4 pb-6 pt-2">
      <ChatInput
        chatId={chatId}
        slug={slug}
        onSend={(text, attachments) => { void send(text, attachments) }}
        disabled={!state || preparing}
        extracting={preparing}
        isStreaming={state?.streaming ?? false}
        onCancel={() => sendToComposer({ type: "composer-abort", sessionId })}
        toolsMenu={false}
        placeholder="Message omp"
      />
      {error && <p role="alert" className="mx-auto mt-1 max-w-2xl px-1 text-xs text-danger">{error}</p>}
    </div>
  )
}

type DropdownProps = ComponentProps<typeof ModelDropdown>

/** gigachad's model selector for an omp tab: the OMP provider rows, applied to the session through the frame. */
export function OmpModelSelector({ view, models, ...dropdown }: { view: OmpView } & Omit<DropdownProps, "selectedModel" | "onSelect">) {
  const { state, sendToComposer } = useComposerFor(view)
  if (view.kind === "settings") return null
  const select = (selectorId: string) => {
    const model = ompModelOf(selectorId)
    if (model && state) sendToComposer({ type: "composer-set-model", sessionId: state.sessionId, ...model })
  }
  return (
    <ModelDropdown
      {...dropdown}
      models={models && ompSelectorModels(models)}
      selectedModel={state?.model ? selectorModelId(state.model) : ""}
      onSelect={select}
    />
  )
}
