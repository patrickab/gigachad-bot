"use client";

import { Children, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CopyAction, TurnCard, TurnHeader, TurnPreview, TurnRegion, TurnSurface } from "@gigachad/chat-ui";
import type { AgentMessage } from "@/lib/types";
import { summarizeTurn } from "./turn-summary";

export interface QaTurnProps {
  /** The whole committed transcript; the turn is `[userIndex, endIndex)`, its typed prompt first. */
  messages: readonly AgentMessage[];
  userIndex: number;
  endIndex: number;
  /** The run is still going: the card stays expanded, without actions. */
  live: boolean;
  /** The prompt, rendered by upstream. */
  question: ReactNode;
  /** The rest of the turn, rendered by upstream. */
  children?: ReactNode;
}

/** Stable across history loads and render windows: the prompt's timestamp, else its index. */
export function qaTurnId(messages: readonly AgentMessage[], userIndex: number): string {
  const { timestamp } = messages[userIndex] as { timestamp?: number };
  return timestamp === undefined ? `index-${userIndex}` : String(timestamp);
}

/** Expand/collapse choices the reader made, by session and turn, so they survive the card leaving the render window. */
const expansionChoices = new Map<string, boolean>();
const MAX_REMEMBERED_CHOICES = 2000;

function rememberChoice(key: string, expanded: boolean) {
  expansionChoices.delete(key);
  expansionChoices.set(key, expanded);
  if (expansionChoices.size > MAX_REMEMBERED_CHOICES) expansionChoices.delete(expansionChoices.keys().next().value as string);
}

/** The open session, as AppShell keeps it in `?session=`; transcripts render client-side only. */
function currentSessionId(): string {
  return new URLSearchParams(window.location.search).get("session") ?? "";
}

/** Copy for one side of the turn, none while it runs. */
function copyAction(text: string, subject: "question" | "answer") {
  return text.trim() ? [<CopyAction key="copy" text={text} label={`Copy ${subject} as markdown`} />] : undefined;
}

/**
 * One user turn as gigachad-bot's shared Q&A card (`@gigachad/chat-ui`): a collapsed header over a scrollable preview of the answer, or the whole turn.
 *
 * - Collapsed: the question's upstream node is not mounted, marker elements keep `data-message-index` for ChatMinimap's lookup.
 *   The preview shows the answer's upstream nodes, tools stay in omp-web's own "Process details" rows
 * - Expanded: the question and the answer's nodes untouched, upstream's per-message fork/edit stay inside; the card owns copy
 * - The newest turn starts expanded, older ones collapsed; the reader's choice sticks per session and turn
 * - A running turn stays expanded and has no actions
 */
export function QaTurn({ messages, userIndex, endIndex, live, question, children }: QaTurnProps) {
  const summary = useMemo(() => summarizeTurn(messages, userIndex, endIndex), [messages, userIndex, endIndex]);
  const choiceKey = `${currentSessionId()}:${qaTurnId(messages, userIndex)}`;
  const [choice, setChoice] = useState(() => expansionChoices.get(choiceKey));
  const [seenKey, setSeenKey] = useState(choiceKey);
  if (seenKey !== choiceKey) {
    setSeenKey(choiceKey);
    setChoice(expansionChoices.get(choiceKey));
  }
  const newest = endIndex === messages.length;
  const expanded = live || (choice ?? newest);

  const toggleRef = useRef<HTMLButtonElement>(null);
  const restoreFocusRef = useRef(false);
  useEffect(() => {
    if (!restoreFocusRef.current) return;
    restoreFocusRef.current = false;
    toggleRef.current?.focus();
  }, [expanded]);
  const toggle = () => {
    restoreFocusRef.current = true;
    rememberChoice(choiceKey, !expanded);
    setChoice(!expanded);
  };

  const anchors = useMemo(() => {
    const indices: number[] = [];
    for (let idx = userIndex; idx < endIndex; idx++) {
      if (messages[idx].role === "user" || messages[idx].role === "assistant") indices.push(idx);
    }
    return indices;
  }, [messages, userIndex, endIndex]);

  const hasAnswer = Children.toArray(children).length > 0;
  if (!expanded) {
    return (
      <TurnCard expanded={false}>
        {anchors.map((idx) => <span key={idx} className="cf-turn-anchor" data-message-index={idx} aria-hidden="true" />)}
        <TurnHeader label={summary.label} onExpand={toggle} ref={toggleRef} />
        <TurnPreview tools={[]}>
          {hasAnswer ? <div className="gc-turn-content">{children}</div> : null}
        </TurnPreview>
      </TurnCard>
    );
  }

  return (
    <TurnCard expanded>
      <TurnRegion
        role="question"
        actions={live ? undefined : copyAction(summary.question, "question")}
        onCollapse={live ? undefined : toggle}
        collapseRef={toggleRef}
      >
        {question}
      </TurnRegion>
      {hasAnswer && (
        <TurnSurface>
          <TurnRegion role="answer" tail={summary.endsWithText ? "text" : "activity"} actions={live ? undefined : copyAction(summary.answer, "answer")}>
            {children}
          </TurnRegion>
        </TurnSurface>
      )}
    </TurnCard>
  );
}
