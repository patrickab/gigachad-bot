/**
 * Contract `SessionInfo` (C5): a Host's session metadata plus pin, profile (G5) and live status.
 *
 * - Consumes: `Host` session metadata and private state, `LiveSessions` for live status
 * - Provides: `SessionInfo`, `sessionInfo`, `listSessions`
 */
import type { FactoryState, Host, Profile, SessionMeta, SessionUsage } from "../host/index.ts";
import type { LiveSessions } from "../supervisor/index.ts";
import { cachedUntil } from "./cache.ts";

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
  /** Present when the session is a factory task: its last `factory-state` entry. */
  factory?: FactoryState;
  /** What its models used, present once one answered. */
  usage?: SessionUsage;
  /** Until when the prompt cache is likely warm (ISO), present when the last answer read or wrote one. */
  cachedUntil?: string;
};

/** Live status under either path form: a new session is keyed before its file exists, a resumed one by real path. */
function liveStatusOf(meta: SessionMeta, live: LiveSessions): LiveStatus | undefined {
  return live.liveStatus(meta.sessionPath) ?? (meta.realPath === meta.sessionPath ? undefined : live.liveStatus(meta.realPath));
}

/**
 * The cache retention `meta`'s Pi ran with: the one pi-host marked when it started it, `short` for a factory task (agent-factory
 * starts tasks so), else pi-host's own, which terminal sessions share through the same environment.
 */
function retentionOf(meta: SessionMeta, marked: string | undefined): string | undefined {
  return marked ?? (meta.factory ? "short" : process.env.PI_CACHE_RETENTION);
}

async function infoOf(host: Host, meta: SessionMeta, live: LiveSessions): Promise<SessionInfo> {
  const { pinned, profile, cacheRetention } = await host.readState(meta.sessionPath);
  const { realPath: _realPath, lastCache, ...fields } = meta;
  return {
    ...fields,
    machine: host.machine,
    live: liveStatusOf(meta, live) ?? null,
    pinned,
    profile: profile ?? "coding",
    ...(lastCache ? { cachedUntil: cachedUntil(lastCache, retentionOf(meta, cacheRetention)) } : {}),
  };
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
