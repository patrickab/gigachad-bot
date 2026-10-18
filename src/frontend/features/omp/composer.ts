import type { Attachment, ModelsResponse, OmpView } from "@/lib/domain/types"
import { OMP_LITELLM_ID } from "@/lib/models"
import { buildHiddenContent, composeUserMessage, isImageAttachment } from "@/features/chat/attachments"
import type { ComposerImage, FrameMessage, ModelRef } from "./protocol"

export type ComposerState = Extract<FrameMessage, { type: "composer" }>

const OMP_MODEL_SOURCE = "omp"

/** The frame's composer state when it belongs to `view`, else null (frame elsewhere, Settings, still loading). */
export function composerFor(view: OmpView, composer: ComposerState | null): ComposerState | null {
  if (!composer || view.kind === "settings") return null
  const sessionId = view.kind === "session" ? view.sessionId : null
  return composer.sessionId === sessionId ? composer : null
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
