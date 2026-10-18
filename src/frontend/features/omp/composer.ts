import type { Attachment, ModelsResponse, OmpView } from "@/lib/domain/types"
import { OMP_LITELLM_ID } from "@/lib/models"
import { buildHiddenContent, composeUserMessage, isImageAttachment } from "@/features/chat/attachments"
import type { ComposerImage, FrameMessage, ModelRef } from "./protocol"

export type ComposerState = Extract<FrameMessage, { type: "composer" }>
export type ComposerRestore = Extract<FrameMessage, { type: "composer-restore" }>

const OMP_MODEL_SOURCE = "omp"

/** Whether a composer message for `sessionId` belongs to `view`. The new-session screen has no session. */
export function isComposerOf(view: OmpView, sessionId: string | null): boolean {
  if (view.kind === "settings") return false
  return sessionId === (view.kind === "session" ? view.sessionId : null)
}

/** The frame's composer state when it belongs to `view`, else null (frame elsewhere, Settings, still loading). */
export function composerFor(view: OmpView, composer: ComposerState | null): ComposerState | null {
  return composer && isComposerOf(view, composer.sessionId) ? composer : null
}

/** Draft after a restore, mirror of omp-web's `mergeRecoveredText` (`lib/draft-store.ts`): later restores replace the block an earlier one put in front. */
export function mergeRestoredText(draft: string, { text, replace }: Pick<ComposerRestore, "text" | "replace">): string {
  if (!text) return draft
  if (replace && draft.startsWith(replace.lead)) return text + draft.slice(replace.lead.length)
  const lead = replace ? replace.fallback : text
  return draft ? `${lead}\n\n${draft}` : lead
}

/** Restored omp image as a file for the composer's upload path. */
export function composerImageFile(image: ComposerImage, index: number): File {
  const bytes = Uint8Array.from(atob(image.data), (character) => character.charCodeAt(0))
  const extension = image.mimeType.split("/")[1] || "png"
  return new File([bytes], `restored-image-${index + 1}.${extension}`, { type: image.mimeType })
}

/** Selector models for omp tabs: only the providers picked from omp's logins, whose models are omp `provider/modelId` ids. */
export function ompSelectorModels(models: ModelsResponse): ModelsResponse {
  return { ...models, ollama: [], providers: models.providers.filter((provider) => provider.source === OMP_MODEL_SOURCE) }
}

/** Selector id of an omp model: `litellm_proxy/anthropic/claude-opus-5`. */
export function selectorModelId(model: ModelRef): string {
  return `${OMP_LITELLM_ID}/${model.provider}/${model.modelId}`
}

/** The omp model a selector id names, null for ids outside the omp rows. */
export function ompModelOf(selectorId: string): ModelRef | null {
  const prefix = `${OMP_LITELLM_ID}/`
  if (!selectorId.startsWith(prefix)) return null
  const id = selectorId.slice(prefix.length)
  const slash = id.indexOf("/")
  if (slash <= 0 || slash === id.length - 1) return null
  return { provider: id.slice(0, slash), modelId: id.slice(slash + 1) }
}

async function base64Image(attachment: Attachment): Promise<ComposerImage> {
  const blob = await (await fetch(attachment.url)).blob()
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(new Error(`Could not read ${attachment.name}`))
    reader.readAsDataURL(blob)
  })
  return { data: dataUrl.slice(dataUrl.indexOf(",") + 1), mimeType: blob.type || attachment.mime }
}

/**
 * omp prompt for a gigachad composer submit.
 *
 * - Images go as omp images (base64)
 * - Documents go into the text the way gigachad chats send them, parsed (PDF) through `parse` when needed
 */
export async function ompPrompt(
  text: string,
  attachments: Attachment[],
  parse: (attachments: Attachment[]) => Promise<Attachment[]>,
): Promise<{ text: string; images: ComposerImage[] }> {
  const active = attachments.map((attachment) => ({ ...attachment, active: true }))
  const images = await Promise.all(active.filter(isImageAttachment).map(base64Image))
  const documents = active.filter((attachment) => !isImageAttachment(attachment))
  if (documents.length === 0) return { text, images }
  const hiddenContent = buildHiddenContent(await parse(documents))
  return { text: composeUserMessage(text, hiddenContent || undefined), images }
}
