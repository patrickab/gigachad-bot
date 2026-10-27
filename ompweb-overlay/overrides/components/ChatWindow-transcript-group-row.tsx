"use client";

import type { ReactNode } from "react";
import {
  renderTranscriptGroupRow as renderUpstreamTranscriptGroupRow,
  type TranscriptGroupRow,
  type TranscriptGroupRowContext,
} from "../../../components/ChatWindow-transcript-group-row";
import { MessageView } from "../../../components/MessageView";
import type { AgentMessage, UserMessage } from "@/lib/types";
import { isEmbedded } from "../../components/EmbedMode";
import { QaTurn, qaTurnId } from "../../components/QaTurn";
import { isQaTurn, skillInvocation } from "../../components/turn-summary";

export * from "../../../components/ChatWindow-transcript-group-row";

/** A skill turn's question: what the user typed, shown like a typed prompt. The skill body stays out, it instructs the agent. Not editable, omp edits user messages only. */
function skillQuestion(messages: readonly AgentMessage[], userIndex: number, invocation: string): ReactNode {
  const { timestamp } = messages[userIndex] as { timestamp?: number };
  const message: UserMessage = { role: "user", content: invocation, timestamp: timestamp ?? 0 };
  return <MessageView key={`skill-question-${userIndex}`} message={message} />;
}

/**
 * Upstream's turn nodes; embedded, a typed prompt's or a skill invocation's turn sits in one Q&A card.
 *
 * - Relies on upstream rendering `renderMessage(row.userIndex)` first, in the live and the folded branch alike: that node is the question
 * - Messages load client-side after hydration, so reading the embed class here never mismatches server markup
 * - Standalone, or another anchor (compaction, branch summary): upstream unchanged
 */
export function renderTranscriptGroupRow(row: TranscriptGroupRow, context: TranscriptGroupRowContext): ReactNode[] {
  const rendered = renderUpstreamTranscriptGroupRow(row, context);
  const { messages, live } = context;
  if (!isEmbedded() || !isQaTurn(messages, row.userIndex)) return rendered;
  const [upstreamQuestion, ...answer] = rendered;
  const invocation = skillInvocation(messages[row.userIndex]);
  const question = invocation === null ? upstreamQuestion : skillQuestion(messages, row.userIndex, invocation);
  return [
    <QaTurn key={`turn-${qaTurnId(messages, row.userIndex)}`} messages={messages} userIndex={row.userIndex} endIndex={row.endIndex} live={live} question={question}>
      {answer}
    </QaTurn>,
  ];
}
