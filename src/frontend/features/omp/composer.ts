import type { Attachment, ModelsResponse, OmpView } from "@/lib/domain/types"
import { blobToDataUrl } from "@/lib/utils"
import { OMP_LITELLM_ID } from "@/lib/models"
import { fuzzyScore } from "@/lib/fuzzy"
import { buildHiddenContent, composeUserMessage, isImageAttachment } from "@/features/chat/attachments"
import type { ComposerImage, FrameMessage, ModelRef } from "@gigachad/chat-ui/protocol"

export type ComposerState = Extract<FrameMessage, { type: "composer" }>
export type ComposerRestore = Extract<FrameMessage, { type: "composer-restore" }>
export type ComposerCommands = Extract<FrameMessage, { type: "composer-commands" }>

/** Command prefix of a skill: `/skill:<name>`. */
export const SKILL_PREFIX = "skill:"

const OMP_MODEL_SOURCE = "omp"

/** Whether a composer message for `sessionId` belongs to `view`. The new-session screen has no session. */
export function isComposerOf(view: OmpView, sessionId: string | null): boolean {
  if (view.kind === "settings") return false
  return sessionId === (view.kind === "session" ? view.sessionId : null)
}

/** Query of the `/` menu: what follows `/` while the whole draft is one `/word`, null otherwise (omp-web opens its menu alike). */
export function slashQueryOf(text: string): string | null {
  return text.startsWith("/") && !/\s/.test(text.slice(1)) ? text.slice(1).toLowerCase() : null
}

/** Whether the menu offers `/skill:` for `query`: while typing towards it (`/`, `/sk`). */
export function offersSkills(query: string): boolean {
  return SKILL_PREFIX.startsWith(query)
}

/** Skill name typed after `/skill:`, null without the full prefix. With it the menu lists skills instead of commands. */
export function skillQueryOf(query: string): string | null {
  return query.startsWith(SKILL_PREFIX) ? query.slice(SKILL_PREFIX.length) : null
}

/** Items whose name matches `query`, best first. No query lists them by name. */
export function rankByName<Item extends { name: string }>(items: Item[], query: string): Item[] {
  if (!query) return [...items].sort((left, right) => left.name.localeCompare(right.name))
  return items
    .map((item) => ({ item, score: fuzzyScore(item.name, query) }))
    .filter((ranked): ranked is { item: Item; score: number } => ranked.score !== null)
    .sort((left, right) => right.score - left.score)
    .map(({ item }) => item)
}

/** An option label as a menu row: a path shows its last segment, the folder beside it (`./scripts/a.sh` -> `a.sh`, `./scripts`). */
export function optionParts(option: string): { name: string; lead: string } {
  const slash = option.lastIndexOf("/")
  return slash === -1 ? { name: option, lead: "" } : { name: option.slice(slash + 1), lead: option.slice(0, slash) }
}

/** Options matching `query`, best first: a match in the last path segment outranks the same match in the folder. No query keeps their order. */
export function rankOptions(options: string[], query: string): string[] {
  const trimmedQuery = query.trim()
  if (!trimmedQuery) return options
  return options
    .map((option) => {
      const byName = fuzzyScore(optionParts(option).name, trimmedQuery)
      return { option, score: Math.max(byName === null ? -Infinity : byName + 1, fuzzyScore(option, trimmedQuery) ?? -Infinity) }
    })
    .filter(({ score }) => score > -Infinity)
    .sort((left, right) => right.score - left.score)
    .map(({ option }) => option)
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

/** Shell mode note like omp-web's input box: `!cmd` sends the output to the model, `!!cmd` keeps it local. Attachments make it a prompt. */
export function shellHint(text: string, attachmentCount: number): string | null {
  const draft = text.trimStart()
  if (attachmentCount > 0 || !draft.startsWith("!")) return null
  return draft.startsWith("!!") ? "Shell · output stays local" : "Shell · output sent to the model"
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
  const dataUrl = await blobToDataUrl(blob).catch(() => { throw new Error(`Could not read ${attachment.name}`) })
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
  return { text: composeUserMessage(text, hiddenContent), images }
}
