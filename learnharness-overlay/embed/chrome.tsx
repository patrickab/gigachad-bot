"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore, type ComponentProps, type ReactNode, type RefObject } from "react";
import { createPortal } from "react-dom";
import { usePathname, useRouter } from "next/navigation";
import {
  LEARN_CHROME_KEYS,
  type LearnActivityItem,
  type LearnChromeKey,
  type LearnFrameMessageOf,
  type LearnPanel,
  type LearnSessionKind,
  type LearnSessionStatus,
  type LearnShellMessageOf,
} from "@gigachad/chat-ui/learn-protocol";
import { ActivitySection, toBlobs, useActivity, type ActivityBlob } from "@/components/activity-dock";
import { CoverageMenuBlock, CoverageRingDetail, fetchGraphRead, type SessionCoverage } from "@/components/extraction-coverage";
import { usePopoverDismiss } from "@/components/popover-dismiss";
import { useServerEvent, useServerResync } from "@/components/server-events";
import { useSettingsDialog } from "@/components/settings/settings-dialog.tsx";
import { kbdEventForAction, type ChatKind, type ChatListItem, type JobKind, type KeyAction } from "@/src/core";
import { AppShell as UpstreamAppShell, SidebarToggle as UpstreamSidebarToggle } from "../../components/sidebar";
import { onShellMessage, postToFrameShell } from "./bridge";
import { isShellDrive, useShellDrive } from "./drive";
import { isEmbedded } from "./mode";
import { sessionIdOf } from "./routes";
import { LANDING_SECTION_SELECTOR } from "./settings";

/**
 * The shell chrome's side in the frame. In drive mode the shell's sidebar and learn tabs stand in for upstream's
 * sessions sidebar and session title bar, which stay mounted but hidden (`overrides/components/sidebar.tsx`,
 * `overrides/components/top-bar.tsx`, `theme/chrome.css`). What they showed reaches the shell from upstream's own sources,
 * what they did the shell asks for and upstream's own UI does:
 *
 * - `session-meta`: the session list the sidebar reads (`GET /api/chats`, refetched on the same server events)
 * - `session-plan`: whether the open session's title bar shows its Plan chip
 * - `activity`: the "Background agents" feed (`useActivity`, folded by `toBlobs` like the section's rows)
 * - `modal`: whether a modal dialog is open (`[role=dialog][aria-modal=true]`, every upstream dialog sets both)
 * - `chrome-key`: the hidden chrome's shortcuts pressed in the frame, the shell's chrome acts on them
 * - `open-settings`: upstream's settings dialog (`useSettingsDialog`), always on a visible section
 * - `open-panel`: upstream's own Plan popup, coverage block and "Background agents" section
 *
 * The sidebar override's components live here (`AppShellFrame`, `DriveSidebarToggle`): the override is a shared module
 * (see its header), so its client code needs a client module that reaches upstream's sidebar by a relative import.
 */

// ─── session-meta ────────────────────────────────────────────────────────────

export type SessionMeta = LearnFrameMessageOf<"session-meta">;

/** Upstream's session kinds as the protocol names them. A new upstream kind fails `tsc` until it is mapped. */
const SESSION_KIND = { teach: "teach", practice: "practice" } as const satisfies Record<ChatKind, LearnSessionKind>;

/** Whether the title bar shows the kind's phase chip: practice sessions show a Practice tag instead (`chat.tsx`). */
const SHOWS_PHASE = { teach: true, practice: false } as const satisfies Record<ChatKind, boolean>;

type MetaSource = Pick<ChatListItem, "id" | "title" | "kind" | "phase" | "busy" | "compacting" | "pendingQuestion" | "pendingProblem" | "updatedAt">;

/** The sidebar row's state line (`sidebar.tsx` `sessionMeta`), most urgent first, null when it shows only the time. */
export function sessionStatusOf(chat: Pick<MetaSource, "kind" | "busy" | "compacting" | "pendingQuestion" | "pendingProblem">): LearnSessionStatus | null {
  if (chat.compacting) return "compacting";
  if (chat.pendingQuestion) return "waiting";
  if ((chat.kind ?? "teach") === "practice" && chat.pendingProblem) return "problem";
  return chat.busy ? "working" : null;
}

/** What the session's sidebar row and title bar show: title, kind, phase (the summary's is the header's, `phaseAt3Lh`), state, age. */
export function sessionMetaOf(chat: MetaSource): SessionMeta {
  const kind: ChatKind = chat.kind ?? "teach";
  return {
    type: "session-meta",
    sessionId: chat.id,
    title: chat.title,
    kind: SESSION_KIND[kind] ?? null,
    phase: SHOWS_PHASE[kind] ? chat.phase : null,
    status: sessionStatusOf(chat),
    updatedAt: chat.updatedAt,
  };
}

const sameMeta = (a: SessionMeta, b: SessionMeta) =>
  a.title === b.title && a.kind === b.kind && a.phase === b.phase && a.status === b.status && a.updatedAt === b.updatedAt;

/** The metas of *chats* that differ from *posted*, which is updated to them. */
export function changedMetas(posted: Map<string, SessionMeta>, chats: readonly MetaSource[]): SessionMeta[] {
  const changed: SessionMeta[] = [];
  for (const chat of chats) {
    const meta = sessionMetaOf(chat);
    const last = posted.get(meta.sessionId);
    if (last && sameMeta(last, meta)) continue;
    posted.set(meta.sessionId, meta);
    changed.push(meta);
  }
  return changed;
}

/** The sessions of *posted* that *chats* no longer lists (deleted), dropped from *posted*. */
export function removedSessions(posted: Map<string, SessionMeta>, chats: readonly Pick<ChatListItem, "id">[]): string[] {
  const listed = new Set(chats.map((chat) => chat.id));
  const removed = [...posted.keys()].filter((id) => !listed.has(id));
  removed.forEach((id) => posted.delete(id));
  return removed;
}

/** Announced list fetches ride upstream's sidebar debounce (the S4 law). */
const LIST_DEBOUNCE_MS = 250;

/** Upstream's session list (`GET /api/chats`, the sidebar's source), null when the read failed. */
async function loadChats(): Promise<ChatListItem[] | null> {
  const response = await fetch("/api/chats", { cache: "no-store" });
  if (!response.ok) return null;
  const { chats } = (await response.json()) as { chats?: ChatListItem[] };
  return Array.isArray(chats) ? chats : null;
}

/** Posts the session metas that changed since the last post. */
export type SessionMetaPoster = {
  /** Read the session list and post every changed meta. One read at a time, a refresh asked meanwhile runs after it. */
  refresh(): Promise<void>;
  /** Upstream's same-tab rename (`lh:chat-renamed`), posted before the server's `chats-changed` arrives. */
  renamed(sessionId: string, title: string): void;
};

/**
 * *post* every changed meta, *removed* the sessions a read no longer lists: deleted, by the shell's session menu
 * (through gigachad's backend) or elsewhere.
 */
export function createSessionMetaPoster(
  post: (meta: SessionMeta) => void,
  load: () => Promise<MetaSource[] | null> = loadChats,
  removed: (sessionIds: string[]) => void = () => {},
): SessionMetaPoster {
  const posted = new Map<string, SessionMeta>();
  let inFlight = false;
  let again = false;
  const refresh = async (): Promise<void> => {
    if (inFlight) {
      again = true;
      return;
    }
    inFlight = true;
    try {
      const chats = await load();
      if (chats) {
        const gone = removedSessions(posted, chats);
        changedMetas(posted, chats).forEach(post);
        if (gone.length > 0) removed(gone);
      }
    } catch {
      // a transient failure keeps the last metas, the next event reads again
    } finally {
      inFlight = false;
      if (again) {
        again = false;
        await refresh();
      }
    }
  };
  const renamed = (sessionId: string, title: string) => {
    const last = posted.get(sessionId);
    if (!last || last.title === title) return;
    const next = { ...last, title };
    posted.set(sessionId, next);
    post(next);
  };
  return { refresh, renamed };
}

/**
 * Post `session-meta` for every session whose row or title bar changed, on the sidebar's own triggers. A session deleted
 * outside the frame gets upstream's delete aftermath (`sidebar.tsx` `onDeleted`): the graph view refetches
 * (`lh:graph-revision`) and its open page goes home.
 */
function useSessionMeta() {
  const router = useRouter();
  const pathname = usePathname();
  const shown = useRef(pathname);
  shown.current = pathname;
  const poster = useRef<SessionMetaPoster | null>(null);
  poster.current ??= createSessionMetaPoster(postToFrameShell, loadChats, (sessionIds) => {
    window.dispatchEvent(new CustomEvent("lh:graph-revision", { detail: {} }));
    const open = sessionIdOf(shown.current);
    if (open !== null && sessionIds.includes(open)) router.push("/");
  });
  const { refresh, renamed } = poster.current;
  const debounce = useRef<number | undefined>(undefined);

  const schedule = () => {
    if (!isEmbedded()) return;
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => void refresh(), LIST_DEBOUNCE_MS);
  };
  useServerEvent("chats-changed", schedule);
  useServerEvent("turn-started", schedule);
  useServerEvent("turn-settled", schedule);
  useServerEvent("compaction-changed", schedule);
  useServerResync(() => {
    if (isEmbedded()) void refresh();
  });

  useEffect(() => {
    if (!isEmbedded()) return;
    void refresh();
    const onRenamed = (event: Event) => {
      const { id, title } = (event as CustomEvent<{ id: string; title: string }>).detail;
      renamed(id, title);
    };
    // The open session's turn flips at once (`lh:turn-state`), the sidebar's same-tab instant: read the list now.
    const onTurnState = () => void refresh();
    window.addEventListener("lh:chat-renamed", onRenamed);
    window.addEventListener("lh:turn-state", onTurnState);
    return () => {
      window.removeEventListener("lh:chat-renamed", onRenamed);
      window.removeEventListener("lh:turn-state", onTurnState);
      window.clearTimeout(debounce.current);
    };
  }, [refresh, renamed]);
}

// ─── activity ────────────────────────────────────────────────────────────────

/** The section's row label per sub-agent kind (`activity-dock.tsx` `ActivityBlobRow`). A new kind fails `tsc`. */
const JOB_LABEL = {
  research: "Researcher",
  demo: "Demo builder",
  visual: "Figure maker",
  pdf: "PDF reader",
} as const satisfies Record<JobKind, string>;

/** One "Background agents" row as the section shows it: the session title, then the activity line or the process label. */
export function activityItemOf(blob: ActivityBlob): LearnActivityItem {
  const process = blob.process === "extraction" ? `Extraction — ${blob.phase ?? "running"}` : blob.jobKind ? JOB_LABEL[blob.jobKind] : "Sub-agent";
  return { id: blob.id, label: blob.title, status: blob.activity || process, sessionId: blob.chatId || null };
}

/** Post `activity` whenever the rows the hidden section lists change. */
function useActivityPost() {
  const { feed } = useActivity();
  const posted = useRef<string | null>(null);
  useEffect(() => {
    if (!isEmbedded()) return;
    const items = toBlobs(feed).map(activityItemOf);
    const key = JSON.stringify(items);
    if (key === posted.current) return;
    posted.current = key;
    postToFrameShell({ type: "activity", items });
  }, [feed]);
}

// ─── settings and modal ──────────────────────────────────────────────────────

/** Upstream's settings nav and its current item (`settings-dialog.tsx`). */
const SETTINGS_NAV = '[data-testid="settings-nav"]';
const CURRENT_SECTION = '[aria-current="true"]';

/** Parts of `document` the settings guard and modal tracker query. */
type ChromeDocument = { querySelector(selector: string): { querySelector(selector: string): unknown; click?(): void } | null };

/**
 * An open settings dialog showing no section (a section name upstream does not know, from `open-settings`) is moved to
 * the landing section, true when it did. The hidden Providers section is `./settings`'s guard.
 */
export function leaveUnknownSection(doc: ChromeDocument): boolean {
  const nav = doc.querySelector(SETTINGS_NAV);
  if (!nav || nav.querySelector(CURRENT_SECTION)) return false;
  const landing = doc.querySelector(LANDING_SECTION_SELECTOR);
  landing?.click?.();
  return landing !== null;
}

/** Any open modal dialog: upstream's dialogs all set `role="dialog"` and `aria-modal="true"`. */
const OPEN_MODAL = '[role="dialog"][aria-modal="true"], [role="alertdialog"][aria-modal="true"]';

/** Whether a modal dialog is open in *doc*. */
export function modalOpen(doc: ChromeDocument): boolean {
  return doc.querySelector(OPEN_MODAL) !== null;
}

/** Parts of `document` closing a modal touches. */
type ModalDocument = { querySelector(selector: string): { dispatchEvent(event: Event): boolean } | null };

/**
 * Close the open modal dialog the way Escape inside it does (`close-modal`, the shell's dimmed chrome was clicked):
 * upstream's dialogs close on a document Escape, after their own open menus (the layered dismiss).
 */
export function closeModal(doc: ModalDocument): void {
  doc.querySelector(OPEN_MODAL)?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }));
}

/** Post `modal` on every change and keep the settings dialog on a section, returns the stop. */
export function trackChrome(doc: Document, post: (message: LearnFrameMessageOf<"modal">) => void): () => void {
  let open: boolean | null = null;
  const check = () => {
    leaveUnknownSection(doc);
    const next = modalOpen(doc);
    if (next === open) return;
    open = next;
    post({ type: "modal", open });
  };
  const observer = new MutationObserver(check);
  observer.observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["aria-current"] });
  check();
  return () => observer.disconnect();
}

/** Open upstream's settings dialog on *section*, null = upstream's default (kept off hidden sections by the guards). */
function useOpenSettings() {
  const { open } = useSettingsDialog();
  useEffect(() => onShellMessage("open-settings", ({ section }) => open(section ?? undefined)), [open]);
}

// ─── keyboard ────────────────────────────────────────────────────────────────

/** Upstream shortcuts that act on the hidden sidebar and title bar: in drive mode they go to the shell's chrome instead. */
export const HIDDEN_CHROME_ACTIONS = LEARN_CHROME_KEYS satisfies readonly KeyAction[];

/** The parts of `window` the keyboard swallow uses. */
type SwallowWindow = {
  addEventListener(type: string, listener: (event: Event) => void, options: { capture: boolean }): void;
  removeEventListener(type: string, listener: (event: Event) => void, options: { capture: boolean }): void;
};

/**
 * While the shell drives, swallow the hidden chrome's shortcut events and hand them to the shell (`chrome-key`): its
 * sidebar toggles (⌘B), its learn filter opens (⌘⇧F), its rename of the open session starts (F2). Returns the stop.
 * Capture listeners on the target run before upstream's (`SidebarProvider`, the sessions sidebar, the chat header).
 */
export function swallowHiddenChrome(
  win: SwallowWindow,
  driven: () => boolean = isShellDrive,
  post: (message: LearnFrameMessageOf<"chrome-key">) => void = postToFrameShell,
): () => void {
  const listeners = HIDDEN_CHROME_ACTIONS.map((action: LearnChromeKey) => {
    const listener = (event: Event) => {
      if (!driven()) return;
      event.stopImmediatePropagation();
      post({ type: "chrome-key", action });
    };
    return [kbdEventForAction(action), listener] as const;
  });
  for (const [type, listener] of listeners) win.addEventListener(type, listener, { capture: true });
  return () => {
    for (const [type, listener] of listeners) win.removeEventListener(type, listener, { capture: true });
  };
}

// ─── panels ──────────────────────────────────────────────────────────────────

/**
 * Where each `open-panel` opens, exhaustive: a new panel fails `tsc` until it has a host.
 *
 * - `title-bar`: the hidden title bar's own control is clicked (`useSessionTitleBar`), upstream's popup shows
 * - `chrome-panel`: upstream's component in a floating card (`ChromePanel`)
 */
export const PANEL_HOST = {
  plan: "title-bar",
  coverage: "chrome-panel",
  activity: "chrome-panel",
} as const satisfies Record<LearnPanel, "title-bar" | "chrome-panel">;

type PanelHost = (typeof PANEL_HOST)[LearnPanel];
export type PanelRequest = LearnShellMessageOf<"open-panel"> & { at: number };

/** A session panel waits this long for its session's page (the shell navigates the frame first), then it is dropped. */
export const PANEL_WAIT_MS = 10_000;

let panelRequest: PanelRequest | null = null;
const panelListeners = new Set<() => void>();
const notifyPanels = () => panelListeners.forEach((listener) => listener());

/** Ask for *message*'s panel, replacing any earlier request. */
export function requestPanel(message: LearnShellMessageOf<"open-panel">, now: number = Date.now()): void {
  panelRequest = { ...message, at: now };
  notifyPanels();
}

/** Drop *request* once its host opened it (or gave up), a newer request stays. */
export function settlePanel(request: PanelRequest): void {
  if (panelRequest !== request) return;
  panelRequest = null;
  notifyPanels();
}

/** The open request for *host*, null while there is none. */
export function currentPanel(host: PanelHost): PanelRequest | null {
  return panelRequest && PANEL_HOST[panelRequest.panel] === host ? panelRequest : null;
}

/** Whether a session panel request still waits for its page. */
export function panelLive(request: PanelRequest, now: number = Date.now()): boolean {
  return now - request.at <= PANEL_WAIT_MS;
}

function subscribePanels(listener: () => void): () => void {
  panelListeners.add(listener);
  return () => panelListeners.delete(listener);
}

/** The open panel request for *host*, re-rendering on every change. */
export function usePanelRequest(host: PanelHost): PanelRequest | null {
  return useSyncExternalStore(subscribePanels, () => currentPanel(host), () => null);
}

/** The session's coverage (`GET /api/chats/<id>` `coverage`, the ring's source), null when it cannot be read. */
async function loadCoverage(chatId: string): Promise<SessionCoverage | null> {
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(chatId)}`, { cache: "no-store" });
    if (!response.ok) return null;
    return ((await response.json()) as { coverage?: SessionCoverage | null }).coverage ?? null;
  } catch {
    return null;
  }
}

/**
 * The coverage ring's popover (the exact numbers, only while the graph is on, like the ring) above the session menu's
 * coverage block (state and "Extract now"), both upstream's.
 */
function CoveragePanel({ chatId }: { chatId: string }) {
  const [detail, setDetail] = useState<{ coverage: SessionCoverage; chunkTokens: number } | null>(null);
  useEffect(() => {
    let alive = true;
    void Promise.all([fetchGraphRead(), loadCoverage(chatId)]).then(([graph, coverage]) => {
      if (alive && graph?.enabled && coverage) setDetail({ coverage, chunkTokens: graph.chunkTokens });
    });
    return () => {
      alive = false;
    };
  }, [chatId]);
  return (
    <>
      {detail && (
        <div className="px-2.5 pt-1 pb-2">
          <CoverageRingDetail coverage={detail.coverage} chunkTokens={detail.chunkTokens} />
        </div>
      )}
      <CoverageMenuBlock chatId={chatId} />
    </>
  );
}

/** The floating card's content per `chrome-panel` panel: upstream's own components. */
const CHROME_PANELS: { [Panel in LearnPanel as (typeof PANEL_HOST)[Panel] extends "chrome-panel" ? Panel : never]: (sessionId: string | null) => ReactNode } = {
  coverage: (sessionId) => (sessionId ? <CoveragePanel key={sessionId} chatId={sessionId} /> : null),
  activity: () => <ActivitySection />,
};

/**
 * Upstream's session-menu coverage block ("Extract now" included) or its "Background agents" section in a floating
 * card at the frame's top right, in the session menu's look. An outside press or Escape closes it (`usePopoverDismiss`).
 */
function ChromePanel() {
  const request = usePanelRequest("chrome-panel");
  const root = useRef<HTMLDivElement>(null);
  const close = useCallback(() => {
    if (request) settlePanel(request);
  }, [request]);
  usePopoverDismiss(root, request !== null, close);
  if (!request) return null;
  const content = CHROME_PANELS[request.panel as keyof typeof CHROME_PANELS](request.sessionId);
  return createPortal(
    <div
      ref={root}
      data-cf-chrome-panel=""
      data-lh-menu=""
      data-panel={request.panel}
      className="flex w-72 flex-col gap-0.5 rounded-xl border bg-card p-2 text-xs shadow-xl"
    >
      {content}
    </div>,
    document.body,
  );
}

/** Upstream's session title bar: its phase chip (every session's title bar shows one, `chat.tsx`). */
export const SESSION_TITLE_CONTENT = '[data-testid="phase-chip"]';
const PLAN_CHIP = '[data-testid="plan-chip"]';

/** The parts of an element the title bar marker reads and sets. */
type TitleBarElement = {
  querySelector(selector: string): unknown;
  toggleAttribute(name: string, force: boolean): unknown;
};

/**
 * Mark *root* (a `data-cf-top-bar`) as the session title bar when it holds one's content, true when it does. Keyed to
 * content, not to the page: another top bar on a session page stays visible, and a title bar upstream reshapes stops
 * matching and shows (fail visible).
 */
export function markSessionTitleBar(root: TitleBarElement, sessionId: string | null): boolean {
  const titleBar = sessionId !== null && root.querySelector(SESSION_TITLE_CONTENT) !== null;
  root.toggleAttribute("data-cf-session-title", titleBar);
  return titleBar;
}

/**
 * The session title bar's part in drive mode (`overrides/components/top-bar.tsx`): marks it for hiding, posts whether
 * it shows a Plan chip (`session-plan`) and opens upstream's Plan popup for a `plan` panel by clicking that chip.
 */
export function useSessionTitleBar(root: RefObject<HTMLElement | null>) {
  const sessionId = sessionIdOf(usePathname());
  const request = usePanelRequest("title-bar");
  const posted = useRef<string | null>(null);

  // Every render: upstream re-renders the bar when its chips change, the marker and the plan follow.
  useLayoutEffect(() => {
    const bar = root.current;
    if (!bar || !markSessionTitleBar(bar, sessionId) || sessionId === null) return;
    const available = bar.querySelector(PLAN_CHIP) !== null;
    const key = `${sessionId}:${available}`;
    if (posted.current === key) return;
    posted.current = key;
    postToFrameShell({ type: "session-plan", sessionId, available });
  });

  useEffect(() => {
    const bar = root.current;
    if (!request || request.sessionId !== sessionId || !bar || !markSessionTitleBar(bar, sessionId)) return;
    if (panelLive(request)) bar.querySelector<HTMLElement>(PLAN_CHIP)?.click();
    settlePanel(request);
  }, [request, sessionId, root]);
}

// ─── mounts ──────────────────────────────────────────────────────────────────

/** Mounted once by `root.tsx`, outside upstream's providers: the `modal` post and `close-modal`, the settings guard, the keyboard handover. */
export function ChromeBridge() {
  useEffect(() => {
    if (!isEmbedded()) return;
    const stops = [trackChrome(document, postToFrameShell), onShellMessage("close-modal", () => closeModal(document)), swallowHiddenChrome(window)];
    return () => stops.forEach((stop) => stop());
  }, []);
  return null;
}

/** Inside upstream's `AppShell`, where the server events, the activity feed and the settings dialog provide. */
function AppShellChromeBridge() {
  useSessionMeta();
  useActivityPost();
  useOpenSettings();
  useEffect(() => onShellMessage("open-panel", (message) => requestPanel(message)), []);
  return <ChromePanel />;
}

/** Upstream's sessions sidebar: the `<aside>` holding its "Teaching" new-session entry (`sidebar.tsx`). */
export const SESSIONS_SIDEBAR_CONTENT = '[data-testid="new-session"]';

/** The parts of an element the sessions sidebar marker reads and sets. */
type ShellElement = { readonly children: ArrayLike<{ tagName: string; querySelector(selector: string): unknown; setAttribute(name: string, value: string): void }> };

/**
 * Mark *root*'s sessions sidebar child for hiding, true when it found it. Keyed to its content: another `<aside>` upstream
 * adds to the app frame stays visible, and a sidebar without its new-session entry shows (fail visible).
 */
export function markSessionsSidebar(root: ShellElement): boolean {
  for (const child of Array.from(root.children)) {
    if (child.tagName !== "ASIDE" || child.querySelector(SESSIONS_SIDEBAR_CONTENT) === null) continue;
    child.setAttribute("data-cf-sessions-sidebar", "");
    return true;
  }
  return false;
}

/**
 * `AppShell` of `overrides/components/sidebar.tsx`: upstream's app frame inside a `data-cf-app-shell` marker
 * (`display: contents`, the layout stays upstream's). The sessions sidebar is marked `data-cf-sessions-sidebar` and
 * hidden in drive mode by `theme/chrome.css`: it stays mounted, its list, settings dialog, shortcuts and "Background
 * agents" feed keep running, and `AppShellChromeBridge` posts what it showed.
 */
export function AppShellFrame({ children, ...props }: ComponentProps<typeof UpstreamAppShell>) {
  const root = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const shell = root.current;
    if (!shell) return;
    markSessionsSidebar(shell);
    const observer = new MutationObserver(() => markSessionsSidebar(shell));
    observer.observe(shell, { childList: true });
    return () => observer.disconnect();
  }, []);
  return (
    <div ref={root} data-cf-app-shell="">
      <UpstreamAppShell {...props}>
        <AppShellChromeBridge />
        {children}
      </UpstreamAppShell>
    </div>
  );
}

/** Upstream's props for *Component*, `object` while it takes none (`ComponentProps` reads a no-argument component as `unknown`). */
type PropsOf<Component> = Component extends (props: infer Props) => unknown ? (unknown extends Props ? object : Props) : never;

/** `SidebarToggle` of `overrides/components/sidebar.tsx`: upstream's reopen button, hidden (`theme/chrome.css`) and inert in drive mode. */
export function DriveSidebarToggle(props: PropsOf<typeof UpstreamSidebarToggle>) {
  const drive = useShellDrive();
  return (
    <span data-cf-sidebar-toggle="" inert={drive}>
      <UpstreamSidebarToggle {...props} />
    </span>
  );
}
