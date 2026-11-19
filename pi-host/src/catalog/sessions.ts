/**
 * Contract `SessionInfo` (C5): a Host's session metadata plus pin and live status.
 *
 * - Consumes: `Host` session metadata and private state, `LiveSessions` for live status
 * - Provides: `SessionInfo`, `sessionInfo`, `listSessions`, `liveStatusOf`
 */
import type { Host, SessionMeta } from "../host/index.ts";
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
  parentSession?: string;
  live: LiveStatus | null;
  pinned: boolean;
  archived: boolean;
};

/** Live status under either path form: a new session is keyed before its file exists, a resumed one by real path. */
export function liveStatusOf(meta: SessionMeta, live: LiveSessions): LiveStatus | undefined {
  return live.liveStatus(meta.sessionPath) ?? (meta.realPath === meta.sessionPath ? undefined : live.liveStatus(meta.realPath));
}

async function infoOf(host: Host, meta: SessionMeta, live: LiveSessions): Promise<SessionInfo> {
  const { pinned } = await host.readState(meta.sessionPath);
  const { realPath: _realPath, ...fields } = meta;
  return { ...fields, machine: host.machine, live: liveStatusOf(meta, live) ?? null, pinned };
}

/** `SessionInfo` of a session file, undefined when it is missing or not a Pi session. */
export async function sessionInfo(host: Host, sessionPath: string, live: LiveSessions): Promise<SessionInfo | undefined> {
  const meta = await host.sessionMeta(sessionPath);
  return meta && infoOf(host, meta, live);
}

/** Sessions of `cwd` (every cwd when null), newest first. */
export async function listSessions(host: Host, cwd: string | null, archived: boolean, live: LiveSessions): Promise<SessionInfo[]> {
  return Promise.all((await host.listSessions(cwd, archived)).map((meta) => infoOf(host, meta, live)));
}
