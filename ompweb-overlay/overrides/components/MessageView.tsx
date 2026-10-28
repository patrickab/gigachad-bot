"use client";

import { memo, useMemo, type ComponentProps } from "react";
import { MessageView as UpstreamMessageView } from "../../../components/MessageView";
import type { CustomMessage } from "@/lib/types";
import { EmbeddedMarkdown } from "../../components/EmbeddedMarkdown";
import { useIsEmbedded } from "../../components/useIsEmbedded";
import { compactionLabel, compactionSummary, isCompaction } from "../../components/turn-summary";

export * from "../../../components/MessageView";

type Props = ComponentProps<typeof UpstreamMessageView>;

/**
 * A compaction as its Q&A card's question: what it saved, the summary folded beneath in the shared Markdown renderer.
 *
 * - Built from upstream's data (`details`, `parseCompactionSummary`), not its markup, so `tsc` catches a changed shape
 * - Upstream's method badge and file lists go, the summary is what a reader needs
 */
function CompactionQuestion({ message }: { message: CustomMessage }) {
  const summary = useMemo(() => compactionSummary(message), [message]);
  return (
    <details className="cf-compaction">
      <summary>{compactionLabel(message)}</summary>
      {summary && <EmbeddedMarkdown>{summary}</EmbeddedMarkdown>}
    </details>
  );
}

/** Upstream's message view; embedded, a compaction renders as `CompactionQuestion`, every other message as upstream does. */
export const MessageView = memo(function MessageView(props: Props) {
  const embedded = useIsEmbedded();
  if (embedded && isCompaction(props.message)) return <CompactionQuestion message={props.message} />;
  return <UpstreamMessageView {...props} />;
});
