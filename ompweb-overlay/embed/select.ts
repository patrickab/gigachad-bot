import { useEffect, useEffectEvent, useMemo } from "react";
import type { UiDialogRequest, UiDialogValue } from "@/lib/types";
import type { SelectRequest } from "@gigachad/chat-ui/protocol";
import { onShellMessage, postToShell } from "./channel";

type SelectDialog = Extract<UiDialogRequest, { method: "select" }>;

/**
 * Hand the session's pending extension `select` to the shell's finder, answer it with the shell's pick.
 *
 * - Returns whether the shell owns the request, then upstream's panel must not render it
 * - Keyed by request id: SSE reconnects replay the same request as a fresh object, the finder keeps its query
 */
export function useShellSelect(
  embedded: boolean,
  sessionId: string | null,
  dialog: UiDialogRequest | null,
  respond: (request: UiDialogRequest, value: UiDialogValue | undefined) => unknown,
): boolean {
  const select = embedded && dialog?.method === "select" ? dialog : null;
  const requestId = select ? String(select.id) : null;
  const request = useMemo<SelectRequest | null>(
    () => (select ? { id: String(select.id), title: select.args[0], options: select.args[1].map((item) => (typeof item === "string" ? item : item.label)) } : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one request per id
    [requestId],
  );

  useEffect(() => {
    if (!embedded) return;
    postToShell({ type: "select", sessionId, request });
    return () => postToShell({ type: "select", sessionId, request: null });
  }, [embedded, sessionId, request]);

  const answer = useEffectEvent((answeredSessionId: string | null, id: string, value: string | null) => {
    if (!select || answeredSessionId !== sessionId || id !== String(select.id)) return;
    void respond(select, value ?? undefined);
  });
  useEffect(() => {
    if (!embedded) return;
    return onShellMessage("select-answer", (message) => answer(message.sessionId, message.id, message.value));
  }, [embedded]);

  return select !== null;
}
