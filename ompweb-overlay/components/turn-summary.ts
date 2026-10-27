import type { AgentMessage, AssistantMessage, CustomMessage, UserMessage } from "@/lib/types";

export interface ToolUse {
  name: string;
  count: number;
}

/** What a Q&A card shows and copies, derived from the turn's messages. */
export interface TurnSummary {
  /** The prompt's markdown source. */
  question: string;
  /** One line for the collapsed header: the prompt's first line, else what it attached. */
  label: string;
  /** Every text block the agent wrote in the turn, markdown source, in order. */
  answer: string;
  /** Tools the turn called, by first use. */
  tools: ToolUse[];
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

/** Turns the card shows: anchored by a typed prompt or a skill invocation. A compaction or branch summary is not one, those render as upstream does. */
export function isQaTurn(messages: readonly AgentMessage[], userIndex: number): boolean {
  const anchor = messages[userIndex];
  return (anchor.role === "user" && !anchor.branchSummary) || isSkillPrompt(anchor);
}

/** Summarize turn `[userIndex, endIndex)` of `messages`; `messages[userIndex]` must be a Q&A anchor (`isQaTurn`). */
export function summarizeTurn(messages: readonly AgentMessage[], userIndex: number, endIndex: number): TurnSummary {
  const skill = skillInvocation(messages[userIndex]);
  const { text: question, images } = skill === null ? userBlocks(messages[userIndex] as UserMessage) : { text: skill, images: 0 };
  const answers: string[] = [];
  const counts = new Map<string, number>();
  let lastMessage: AgentMessage | undefined;
  for (let idx = userIndex + 1; idx < endIndex; idx++) {
    const message = messages[idx];
    if (message.role === "toolResult") continue;
    lastMessage = message;
    if (message.role !== "assistant") continue;
    for (const block of (message as AssistantMessage).content ?? []) {
      if (block.type === "text" && block.text.trim()) answers.push(block.text);
      else if (block.type === "toolCall") counts.set(block.toolName, (counts.get(block.toolName) ?? 0) + 1);
    }
  }
  return {
    question,
    label: questionLabel(question, images),
    answer: answers.join("\n\n"),
    tools: [...counts].map(([name, count]) => ({ name, count })),
    endsWithText: endsInText(lastMessage),
  };
}
