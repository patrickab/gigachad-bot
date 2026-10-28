import {
  isGroupAnchor as isUpstreamGroupAnchor,
  planTranscriptRows as planUpstreamTranscriptRows,
  planTurnSegments,
  type PlanOptions,
  type TranscriptRow,
} from "../../../lib/chat-transcript-plan";
import type { AgentMessage } from "../../../lib/types";
import { isEmbedded } from "../../components/EmbedMode";
import { isQaTurn, isSkillPrompt } from "../../components/turn-summary";

export * from "../../../lib/chat-transcript-plan";

/**
 * Upstream's turn anchors; embedded, a skill invocation anchors its turn too.
 *
 * omp records `/skill:name args` as a "skill-prompt" custom message without a user message, upstream would fold
 * the skill's turn into the previous one. ChatWindow's running-turn check imports this, so it agrees with the rows.
 */
export function isGroupAnchor(message: AgentMessage): boolean {
  return isUpstreamGroupAnchor(message) || (isEmbedded() && isSkillPrompt(message));
}

/**
 * Upstream's rows; embedded, a skill invocation starts its own turn and a plain Q&A turn is one `group` row too, so its card wraps the whole turn.
 *
 * Upstream emits a turn without activity as consecutive `message` rows, its anchor first, up to the next anchor or group.
 * Other anchors (branch summary) keep upstream's rows. Messages load client-side after hydration,
 * so reading the embed class here never mismatches server markup.
 */
export function planTranscriptRows(messages: AgentMessage[], options: PlanOptions = {}): TranscriptRow[] {
  const rows = planUpstreamTranscriptRows(messages, options);
  return isEmbedded() ? groupPlainTurns(splitAtSkillPrompts(rows, messages, options), messages, options) : rows;
}

/** Split upstream's group rows where a skill invocation folded into the previous turn. */
function splitAtSkillPrompts(rows: TranscriptRow[], messages: AgentMessage[], options: PlanOptions): TranscriptRow[] {
  return rows.flatMap((row): TranscriptRow[] => {
    if (row.kind !== "group") return [row];
    const starts = [row.userIndex];
    for (let idx = row.userIndex + 1; idx < row.endIndex; idx++) {
      if (isSkillPrompt(messages[idx])) starts.push(idx);
    }
    if (starts.length === 1) return [row];
    return starts.map((start, startIdx) => {
      const end = starts[startIdx + 1] ?? row.endIndex;
      return { kind: "group", userIndex: start, endIndex: end, segments: planTurnSegments(messages, start, end, options) };
    });
  });
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
