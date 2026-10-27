import {
  isGroupAnchor,
  planTranscriptRows as planUpstreamTranscriptRows,
  planTurnSegments,
  type PlanOptions,
  type TranscriptRow,
} from "../../../lib/chat-transcript-plan";
import type { AgentMessage } from "../../../lib/types";
import { isEmbedded } from "../../components/EmbedMode";
import { isQaTurn } from "../../components/turn-summary";

export * from "../../../lib/chat-transcript-plan";

/**
 * Upstream's rows; embedded, a plain Q&A turn is one `group` row too, so its card wraps the whole turn.
 *
 * Upstream emits a turn without activity as consecutive `message` rows, its anchor first, up to the next anchor or group.
 * Other anchors (compaction, branch summary) keep upstream's rows. Messages load client-side after hydration,
 * so reading the embed class here never mismatches server markup.
 */
export function planTranscriptRows(messages: AgentMessage[], options: PlanOptions = {}): TranscriptRow[] {
  const rows = planUpstreamTranscriptRows(messages, options);
  return isEmbedded() ? groupPlainTurns(rows, messages, options) : rows;
}

function groupPlainTurns(rows: TranscriptRow[], messages: AgentMessage[], options: PlanOptions): TranscriptRow[] {
  const grouped: TranscriptRow[] = [];
  for (let rowIdx = 0; rowIdx < rows.length;) {
    const row = rows[rowIdx];
    rowIdx += 1;
    if (row.kind !== "message" || !isQaTurn(messages, row.index)) {
      grouped.push(row);
      continue;
    }
    let endIdx = row.index + 1;
    while (rowIdx < rows.length) {
      const next = rows[rowIdx];
      if (next.kind !== "message" || isGroupAnchor(messages[next.index])) break;
      endIdx = next.index + 1;
      rowIdx += 1;
    }
    grouped.push({ kind: "group", userIndex: row.index, endIndex: endIdx, segments: planTurnSegments(messages, row.index, endIdx, options) });
  }
  return grouped;
}
