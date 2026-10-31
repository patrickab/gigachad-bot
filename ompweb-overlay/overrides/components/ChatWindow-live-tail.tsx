"use client";

import type { ComponentProps } from "react";
import { ChatLiveTail as UpstreamChatLiveTail } from "../../../components/ChatWindow-live-tail";
import { LiveTail } from "../../components/LiveTail";
import { useIsEmbedded } from "../../components/useIsEmbedded";

export * from "../../../components/ChatWindow-live-tail";

/** Upstream's live tail; embedded, inside the newest Q&A card (`LiveTail`). Standalone: upstream unchanged. */
export function ChatLiveTail(props: ComponentProps<typeof UpstreamChatLiveTail>) {
  const tail = <UpstreamChatLiveTail {...props} />;
  return useIsEmbedded() ? <LiveTail>{tail}</LiveTail> : tail;
}
