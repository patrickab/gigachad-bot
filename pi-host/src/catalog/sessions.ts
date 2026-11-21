/**
 * Contract `SessionInfo` (C5): a Host's session metadata plus pin, profile (G5) and live status.
 *
 * - Consumes: `Host` session metadata and private state, `LiveSessions` for live status
 * - Provides: `SessionInfo`, `sessionInfo`, `listSessions`
 */
import type { Host, Profile, SessionMeta } from "../host/index.ts";
import type { LiveSessions } from "../supervisor/index.ts";

export type LiveStatus = { running: boolean; waiting: boolean; attached: number };

export type SessionInfo = {
  sessionId: string;
  sessionPath: string;
  cwd: string;
  machine: string;
  createdAt: string;
  modifiedAt: string;
  firstMessage: string;
  messageCount: number;
  /** Context size after the last answer, 0 before one. */
  tokens: number;
  parentSession?: string;
  live: LiveStatus | null;
  pinned: boolean;
  archived: boolean;
  /** The profile the session was started with, `coding` when its sidecar has no marker. */
  profile: Profile;
};

/** Live status under either path form: a new session is keyed before its file exists, a resumed one by real path. */
function liveStatusOf(meta: SessionMeta, live: LiveSessions): LiveStatus | undefined {
  return live.liveStatus(meta.sessionPath) ?? (meta.realPath === meta.sessionPath ? undefined : live.liveStatus(meta.realPath));
}

async function infoOf(host: Host, meta: SessionMeta, live: LiveSessions): Promise<SessionInfo> {
  const { pinned, profile } = await host.readState(meta.sessionPath);
  const { realPath: _realPath, ...fields } = meta;
  return { ...fields, machine: host.machine, live: liveStatusOf(meta, live) ?? null, pinned, profile: profile ?? "coding" };
}

/** `SessionInfo` of a session file, undefined when it is missing or not a Pi session. */
export async function sessionInfo(host: Host, sessionPath: string, live: LiveSessions): Promise<SessionInfo | undefined> {
  const meta = await host.sessionMeta(sessionPath);
  return meta && infoOf(host, meta, live);
}

/** Sessions of `cwd` (every cwd when null), newest first, only those of `profile` when given. */
export async function listSessions(
  host: Host,
  cwd: string | null,
  archived: boolean,
  live: LiveSessions,
  profile?: Profile,
): Promise<SessionInfo[]> {
  const sessions = await Promise.all((await host.listSessions(cwd, archived)).map((meta) => infoOf(host, meta, live)));
  return profile ? sessions.filter((session) => session.profile === profile) : sessions;
}
