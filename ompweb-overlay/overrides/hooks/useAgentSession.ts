"use client";

import { useEffect, useMemo } from "react";
import { useAgentSession as useUpstreamAgentSession, type UseAgentSessionOptions } from "../../../hooks/useAgentSession";
import { useIsEmbedded } from "../../components/useIsEmbedded";
import { postToShell } from "../../embed/channel";
import { openFileNoticePath } from "../../embed/open-file";

export * from "../../../hooks/useAgentSession";

function isOpenFileNotice(notice: { message: string }): boolean {
  return openFileNoticePath(notice.message) !== null;
}

/**
 * Upstream session, whose open-file notices (`embed/open-file.ts`) open the file in gigachad-bot's file view when embedded.
 *
 * - An extension command reaches the browser only through omp's RPC UI requests, `notify` is the one carrying a value once
 * - The notice never shows, it is dismissed once the shell has the path
 * - Standalone: upstream unchanged, the notice names the file
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

  const shownNotices = useMemo(() => notices.filter((notice) => !isOpenFileNotice(notice)), [notices]);
  return embedded && shownNotices.length !== notices.length ? { ...session, notices: shownNotices } : session;
}
