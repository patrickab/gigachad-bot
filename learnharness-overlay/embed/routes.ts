import type { LearnShellMessage } from "@gigachad/chat-ui/learn-protocol";

const NEW_SESSION_PATH = "/chat/new";
const SESSION_PATH = /^\/chat\/([^/]+)\/?$/;

/** LearnHarness session shown at *pathname*: the id of `/chat/<id>`, null for `/chat/new`, a malformed id and every other page. */
export function sessionIdOf(pathname: string): string | null {
  if (pathname === NEW_SESSION_PATH || pathname === `${NEW_SESSION_PATH}/`) return null;
  const match = SESSION_PATH.exec(pathname);
  if (!match?.[1]) return null;
  try {
    return decodeURIComponent(match[1]);
  } catch {
    return null;
  }
}

type NavigationMessage = Extract<LearnShellMessage, { type: "open-session" | "new-session" | "open-home" }>;

/** Page the shell's navigation message opens. A new session takes upstream's own entries: Teaching `/chat/new`, Practice `?mode=practice`. */
export function pathFor(message: NavigationMessage): string {
  switch (message.type) {
    case "open-session":
      return `/chat/${encodeURIComponent(message.sessionId)}`;
    case "new-session":
      return message.kind === "practice" ? `${NEW_SESSION_PATH}?mode=practice` : NEW_SESSION_PATH;
    case "open-home":
      return "/";
  }
}
