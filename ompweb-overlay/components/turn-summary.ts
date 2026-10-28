import { parseCompactionSummary } from "@/lib/compaction-summary";
import { formatCompactNumber } from "@/lib/format";
import type { AgentMessage, AssistantMessage, CustomMessage, UserMessage } from "@/lib/types";

/** What a Q&A card shows and copies, derived from the turn's messages. */
export interface TurnSummary {
  /** The prompt's markdown source, for a compaction its summary. */
  question: string;
  /** One line for the collapsed header: the prompt's first line, else what it attached, for a compaction what it saved. */
  label: string;
  /** Every text block the agent wrote in the turn, markdown source, in order. */
  answer: string;
  /** The turn ends with agent text, the only tail the card can put actions beside. */
  endsWithText: boolean;
}

const MAX_LABEL_LENGTH = 160;

function userBlocks(message: UserMessage): { text: string; images: number } {
  if (typeof message.content === "string") return { text: message.content, images: 0 };
  let images = 0;
  const texts: string[] = [];
  for (const block of message.content) {
    if (block.type === "image") images += 1;
    else texts.push(block.text);
  }
  return { text: texts.join("\n"), images };
}

function questionLabel(text: string, images: number): string {
  const firstLine = text.split("\n").find((line) => line.trim().length > 0)?.trim();
  if (firstLine) return firstLine.length <= MAX_LABEL_LENGTH ? firstLine : `${firstLine.slice(0, MAX_LABEL_LENGTH)}\u2026`;
  return images > 0 ? (images === 1 ? "Image" : `${images} images`) : "(empty)";
}

/** Whether `message` is the turn's last agent message and ends in text; tool results and hidden notices never do. */
function endsInText(message: AgentMessage | undefined): boolean {
  if (message?.role !== "assistant") return false;
  const blocks = (message as AssistantMessage).content ?? [];
  const last = blocks[blocks.length - 1];
  return last?.type === "text" && last.text.trim().length > 0 && !(message as AssistantMessage).errorMessage?.trim();
}

/** omp records a skill invocation (`/skill:name args`) as this custom message alone, no user message. */
function isSkillPrompt(message: AgentMessage): message is CustomMessage {
  return message.role === "custom" && (message as CustomMessage).customType === "skill-prompt";
}

/** What the user typed to invoke a skill, null for any other message. The skill's body is instructions for the agent, not the question. */
export function skillInvocation(message: AgentMessage): string | null {
  if (!isSkillPrompt(message)) return null;
  const { prompt, name, args } = (message.details ?? {}) as { prompt?: unknown; name?: unknown; args?: unknown };
  if (typeof prompt === "string" && prompt.trim()) return prompt;
  const command = `/skill:${typeof name === "string" ? name : "unknown"}`;
  return typeof args === "string" && args.trim() ? `${command} ${args}` : command;
}

/** omp records a compaction as this custom message, the turn that continues after it is anchored there. */
export function isCompaction(message: AgentMessage): message is CustomMessage {
  return message.role === "custom" && (message as CustomMessage).customType === "compaction";
}

/** The compaction's summary, markdown source, without the trailing file lists. */
export function compactionSummary(message: CustomMessage): string {
  const { content } = message;
  const text = typeof content === "string" ? content : content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
  return parseCompactionSummary(text).body.trim();
}

/** "Conversation compacted · 85k → 44k tokens", the counts when the entry records them. */
export function compactionLabel(message: CustomMessage): string {
  const { tokensBefore, tokensAfter } = (message.details ?? {}) as { tokensBefore?: unknown; tokensAfter?: unknown };
  if (typeof tokensBefore !== "number" || typeof tokensAfter !== "number") return "Conversation compacted";
  return `Conversation compacted · ${formatCompactNumber(tokensBefore)} → ${formatCompactNumber(tokensAfter)} tokens`;
}

/** Turns the card shows: anchored by a typed prompt, a skill invocation or a compaction. A branch summary is not one, it renders as upstream does. */
export function isQaTurn(messages: readonly AgentMessage[], userIndex: number): boolean {
  const anchor = messages[userIndex];
  return (anchor.role === "user" && !anchor.branchSummary) || isSkillPrompt(anchor) || isCompaction(anchor);
}

/** The anchor's markdown source and its one-line label. */
function anchorQuestion(anchor: AgentMessage): { question: string; label: string } {
  if (isCompaction(anchor)) return { question: compactionSummary(anchor), label: compactionLabel(anchor) };
  const skill = skillInvocation(anchor);
  if (skill !== null) return { question: skill, label: questionLabel(skill, 0) };
  const { text, images } = userBlocks(anchor as UserMessage);
  return { question: text, label: questionLabel(text, images) };
}

/** Summarize turn `[userIndex, endIndex)` of `messages`; `messages[userIndex]` must be a Q&A anchor (`isQaTurn`). */
export function summarizeTurn(messages: readonly AgentMessage[], userIndex: number, endIndex: number): TurnSummary {
  const { question, label } = anchorQuestion(messages[userIndex]);
  const answers: string[] = [];
  let lastMessage: AgentMessage | undefined;
  for (let idx = userIndex + 1; idx < endIndex; idx++) {
    const message = messages[idx];
    if (message.role === "toolResult") continue;
    lastMessage = message;
    if (message.role !== "assistant") continue;
    for (const block of (message as AssistantMessage).content ?? []) {
      if (block.type === "text" && block.text.trim()) answers.push(block.text);
    }
  }
  return {
    question,
    label,
    answer: answers.join("\n\n"),
    endsWithText: endsInText(lastMessage),
  };
}
