import type { OpenedWorkbenchChat, PiSessionInfo, WorkbenchChatKind, WorkbenchPdfText } from "@/lib/backend/api"
import type { PiInsertion } from "@/features/pi/PiComposer"
import type { DocumentChat } from "../useDocumentChat"

/** An opened workbench chat of session `id` in `cwd`: new (no messages) unless `messageCount` says otherwise. */
export function openedChat({ id, cwd, kind = "markdown", messageCount = 0, moved = [], pdfText = null, firstMessage = "" }: {
  id: string
  cwd: string
  kind?: WorkbenchChatKind
  messageCount?: number
  moved?: string[]
  pdfText?: WorkbenchPdfText | null
  firstMessage?: string
}): OpenedWorkbenchChat {
  const session: PiSessionInfo = {
    sessionId: id,
    sessionPath: `/sessions/${id}.jsonl`,
    cwd,
    machine: "local",
    createdAt: "2026-01-01T00:00:00Z",
    modifiedAt: "2026-01-01T00:00:00Z",
    firstMessage,
    messageCount,
    live: null,
    pinned: false,
    archived: false,
    profile: "document",
  }
  return { session, link: { active: id, sessionIds: [id], moved }, kind, cwd, pdfText }
}

/** Stand-in for `DocumentChat`: the chat's key, shown session and lead, and the draft insertion. */
export function FakeDocumentChat({ chat, insertion }: { chat: DocumentChat; insertion?: PiInsertion | null }) {
  return <>
    <p>chat {chat.path} {chat.opened?.session.sessionId ?? "none"} {chat.lead ?? "no lead"}</p>
    {insertion && <pre data-testid="draft">{insertion.text}</pre>}
  </>
}
