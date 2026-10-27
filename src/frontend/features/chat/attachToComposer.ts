import type { RefObject } from "react"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { attachReference } from "@/lib/backend/references"

/** Read `path` as a live reference (busy through `track`) and stage it on the composer; a failed read leaves the composer unchanged. */
export async function attachToComposer(chatInputRef: RefObject<ChatInputHandle | null>, track: <T>(work: () => Promise<T>) => Promise<T>, path: string): Promise<void> {
  try {
    chatInputRef.current?.addAttachment(await track(() => attachReference(path)))
  } catch { /* attach failure leaves the input unchanged */ }
}
