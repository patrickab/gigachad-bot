"use client";

import { memo, useEffect, useMemo, useRef, useState, type ComponentProps } from "react";
import { FoldVertical } from "lucide-react";
import { CopyAction, TurnCard, TurnHeader, TurnRegion, TurnSurface } from "@gigachad/chat-ui";
import { MessageView as UpstreamMessageView } from "../../../components/MessageView";
import { parseCompactionSummary } from "@/lib/compaction-summary";
import { formatCompactNumber } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import type { AgentMessage, CustomMessage } from "@/lib/types";
import { EmbeddedMarkdown } from "../../components/EmbeddedMarkdown";
import { useIsEmbedded } from "../../components/useIsEmbedded";

export * from "../../../components/MessageView";

type Props = ComponentProps<typeof UpstreamMessageView>;

function isCompaction(message: AgentMessage): message is CustomMessage {
  return message.role === "custom" && (message as CustomMessage).customType === "compaction";
}

function messageText(content: CustomMessage["content"]): string {
  return typeof content === "string" ? content : content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
}

/** Upstream's "Conversation compacted", plus "85k → 44k tokens" from omp ≥17.4 entries, which record both counts. */
function useCompactionLabel(message: CustomMessage): string {
  const { t, locale } = useI18n();
  const details = (message.details ?? null) as { tokensBefore?: unknown; tokensAfter?: unknown } | null;
  const { tokensBefore, tokensAfter } = details ?? {};
  const title = t("messageView.conversationCompacted");
  if (typeof tokensBefore !== "number" || typeof tokensAfter !== "number") return title;
  const delta = t("messageView.compactionTokenDelta", { before: formatCompactNumber(tokensBefore, locale), after: formatCompactNumber(tokensAfter, locale) });
  return `${title} · ${delta}`;
}

/**
 * A compaction as gigachad's turn card: one collapsed line, expanded the summary like an answer, in the shared Markdown renderer.
 *
 * - Built from upstream's data (`details`, `parseCompactionSummary`), not its markup, so `tsc` catches a changed shape
 * - Upstream's method badge and file lists go, the summary is what a reader needs
 */
function CompactionTurn({ message }: { message: CustomMessage }) {
  const { t } = useI18n();
  const label = useCompactionLabel(message);
  const summary = useMemo(() => parseCompactionSummary(messageText(message.content)).body.trim(), [message.content]);
  const [expanded, setExpanded] = useState(false);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const restoreFocusRef = useRef(false);
  useEffect(() => {
    if (!restoreFocusRef.current) return;
    restoreFocusRef.current = false;
    toggleRef.current?.focus();
  }, [expanded]);
  const toggle = () => {
    restoreFocusRef.current = true;
    setExpanded(!expanded);
  };
  const icon = <FoldVertical size={14} aria-hidden="true" />;

  if (!expanded) {
    return (
      <TurnCard expanded={false}>
        <TurnHeader label={label} icon={icon} onExpand={toggle} ref={toggleRef} />
      </TurnCard>
    );
  }
  return (
    <TurnCard expanded>
      <TurnRegion role="question" icon={icon} onCollapse={toggle} collapseRef={toggleRef} collapseLabel="Collapse compaction summary">
        {label}
      </TurnRegion>
      <TurnSurface>
        <TurnRegion role="answer" actions={summary ? <CopyAction text={summary} label="Copy summary as markdown" /> : undefined}>
          {summary ? <EmbeddedMarkdown>{summary}</EmbeddedMarkdown> : t("messageView.noSummary")}
        </TurnRegion>
      </TurnSurface>
    </TurnCard>
  );
}

/**
 * Upstream's message view; embedded, a compaction renders as a turn card (`CompactionTurn`).
 *
 * - Standalone and every other message: upstream unchanged
 */
export const MessageView = memo(function MessageView(props: Props) {
  const embedded = useIsEmbedded();
  if (embedded && isCompaction(props.message)) return <CompactionTurn message={props.message} />;
  return <UpstreamMessageView {...props} />;
});
