"use client";

import { useEffect, useMemo } from "react";
import { useAgentSession as useUpstreamAgentSession, type UseAgentSessionOptions } from "../../../hooks/useAgentSession";
import { useIsEmbedded } from "../../components/useIsEmbedded";
import { postToShell } from "../../embed/channel";
import { openFileNoticePath } from "../../embed/open-file";
import { useShellSelect } from "../../embed/select";

export * from "../../../hooks/useAgentSession";

function isOpenFileNotice(notice: { message: string }): boolean {
  return openFileNoticePath(notice.message) !== null;
}

/**
 * Upstream session with two extension UI requests handed to the shell when embedded.
 *
 * - Open-file notices (`embed/open-file.ts`) open the file in gigachad-bot's file view and never show. An extension command
 *   reaches the browser only through omp's RPC UI requests, `notify` is the one carrying a value once
 * - A `select` is picked in the shell's fuzzy finder (`embed/select.ts`), upstream's panel gets no dialog to render
 * - Standalone: upstream unchanged
 */
export function useAgentSession(options: UseAgentSessionOptions) {
  const session = useUpstreamAgentSession(options);
  const embedded = useIsEmbedded();
  const { notices, dismissNotice } = session;

  useEffect(() => {
    if (!embedded) return;
    for (const notice of notices) {
      const path = openFileNoticePath(notice.message);
      if (path === null) continue;
      postToShell({ type: "open-file", path });
      dismissNotice(notice.id);
    }
  }, [embedded, notices, dismissNotice]);

  const selectInShell = useShellSelect(embedded, options.session?.id ?? null, session.extensionDialog, session.respondToExtensionUi);

  const shownNotices = useMemo(() => notices.filter((notice) => !isOpenFileNotice(notice)), [notices]);
  if (!embedded) return session;
  return { ...session, notices: shownNotices, extensionDialog: selectInShell ? null : session.extensionDialog };
}
