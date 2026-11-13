"use client";

import { useContext, type ComponentProps } from "react";
import { ChatLiveTail as UpstreamChatLiveTail } from "../../../components/ChatWindow-live-tail";
import type { AgentMessage } from "@/lib/types";
import { LiveTail, ToolActivityHidden } from "../../components/LiveTail";
import { useIsEmbedded } from "../../components/useIsEmbedded";

export * from "../../../components/ChatWindow-live-tail";

type Props = ComponentProps<typeof UpstreamChatLiveTail>;

/** The streaming reply without its thinking and tool call blocks, null when nothing else is left. */
function withoutActivity(message: Partial<AgentMessage> | null): Partial<AgentMessage> | null {
  if (message?.role !== "assistant" || message.errorMessage) return message;
  const content = (message.content ?? []).filter((block) => block.type !== "thinking" && block.type !== "toolCall");
  return content.length > 0 ? { ...message, content } : null;
}

/**
 * Upstream's live tail; embedded, inside the newest Q&A card (`LiveTail`). Standalone: upstream unchanged.
 *
 * - Tool activity hidden: the streaming reply shows its answer only. Each thinking or tool row would flash in the card
 *   until its message commits into "Process details", resizing the card every step. The composer's status card names the step
 */
export function ChatLiveTail(props: Props) {
  const embedded = useIsEmbedded();
  const activityHidden = useContext(ToolActivityHidden);
  if (!embedded) return <UpstreamChatLiveTail {...props} />;
  const streamingMessage = activityHidden ? withoutActivity(props.streamingMessage) : props.streamingMessage;
  return <LiveTail><UpstreamChatLiveTail {...props} streamingMessage={streamingMessage} /></LiveTail>;
}
