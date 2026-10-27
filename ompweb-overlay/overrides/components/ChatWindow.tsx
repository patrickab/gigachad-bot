"use client";

import { useCallback, type ComponentProps } from "react";
import { ChatWindow as UpstreamChatWindow } from "../../../components/ChatWindow";
import { postToShell } from "../../embed/channel";
import { resolveOpenFilePath } from "../../embed/open-file";
import { useIsEmbedded } from "../../components/useIsEmbedded";

export * from "../../../components/ChatWindow";

type Props = ComponentProps<typeof UpstreamChatWindow>;

/**
 * Upstream chat, whose local file links open in gigachad-bot's file view when embedded.
 *
 * - `onOpenFile` reaches every file link of the conversation (markdown, Q&A cards, tool output, /btw), embedded mode hides the file panel it would open
 * - Relative tool paths resolve against the session cwd first, the shell needs absolute ones
 * - Standalone: upstream unchanged
 */
export function ChatWindow(props: Props) {
  const embedded = useIsEmbedded();
  const cwd = props.session?.cwd ?? props.newSessionCwd ?? undefined;
  const openFileInShell = useCallback((path: string) => postToShell({ type: "open-file", path: resolveOpenFilePath(path, cwd) }), [cwd]);
  return <UpstreamChatWindow {...props} onOpenFile={embedded ? openFileInShell : props.onOpenFile} />;
}
