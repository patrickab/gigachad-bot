"use client";

import { useEffect, useRef, type ComponentProps } from "react";
import { BranchNavigator as UpstreamBranchNavigator } from "../../../components/BranchNavigator";
import type { SessionEntry, SessionTreeNode } from "@/lib/types";
import { isEmbedded } from "../../components/EmbedMode";
import { onShellMessage, postToShell } from "../../embed/channel";
import type { BranchNode } from "@gigachad/chat-ui/protocol";

type Props = ComponentProps<typeof UpstreamBranchNavigator>;

const PREVIEW_LENGTH = 80;

function messageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((block) => (block?.type === "text" && typeof block.text === "string" ? block.text : "")).join(" ");
}

function messageRole(entry: SessionEntry): BranchNode["role"] {
  const role = entry.type === "message" && "message" in entry ? entry.message.role : undefined;
  return role === "user" || role === "assistant" ? role : undefined;
}

function messageContent(entry: SessionEntry): unknown {
  return entry.type === "message" && "message" in entry && "content" in entry.message ? entry.message.content : undefined;
}

function toBranchNode(node: SessionTreeNode): BranchNode {
  const { entry } = node;
  const text = messageText(messageContent(entry));
  return {
    id: entry.id,
    type: entry.type,
    role: messageRole(entry),
    preview: text.replace(/\s+/g, " ").trim().slice(0, PREVIEW_LENGTH),
    label: node.label,
    children: node.children.map(toBranchNode),
  };
}

/** Session id the chat shows, `select-leaf` for any other session is stale. */
function shownSessionId(): string | null {
  return new URLSearchParams(window.location.search).get("session");
}

/**
 * Upstream branch navigator, plus the tree for gigachad-bot's sidebar while embedded.
 *
 * - Embedded: upstream's copy sits in the hidden topbar, so the tree and leaf selection go through the bridge
 * - Standalone: upstream unchanged
 */
export function BranchNavigator(props: Props) {
  const { tree, activeLeafId, onLeafChange } = props;
  const onLeafChangeRef = useRef(onLeafChange);
  const lastPostedRef = useRef("");
  useEffect(() => { onLeafChangeRef.current = onLeafChange; }, [onLeafChange]);

  useEffect(() => {
    const sessionId = isEmbedded() ? shownSessionId() : null;
    if (!sessionId) return;
    const message = { type: "branch-tree", sessionId, tree: tree.map(toBranchNode), activeLeafId } as const;
    const serialized = JSON.stringify(message);
    if (serialized === lastPostedRef.current) return;
    lastPostedRef.current = serialized;
    postToShell(message);
  }, [tree, activeLeafId]);

  useEffect(() => {
    if (!isEmbedded()) return;
    return onShellMessage("select-leaf", (message) => {
      if (message.sessionId === shownSessionId()) onLeafChangeRef.current(message.leafId);
    });
  }, []);

  return <UpstreamBranchNavigator {...props} />;
}
