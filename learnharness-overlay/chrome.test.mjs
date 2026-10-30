import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, { tsconfigPaths: true, jsx: true });
// jiti compiles JSX for the classic runtime.
globalThis.React = await import("react");
const h = React.createElement;
const { renderToStaticMarkup } = await import("react-dom/server");
const { AppRouterContext } = await import("next/dist/shared/lib/app-router-context.shared-runtime.js");
const { PathnameContext } = await import("next/dist/shared/lib/hooks-client-context.shared-runtime.js");

const {
  activityItemOf,
  closeModal,
  changedMetas,
  createSessionMetaPoster,
  currentPanel,
  HIDDEN_CHROME_ACTIONS,
  leaveUnknownSection,
  markSessionsSidebar,
  markSessionTitleBar,
  PANEL_HOST,
  PANEL_WAIT_MS,
  panelLive,
  removedSessions,
  requestPanel,
  SESSION_TITLE_CONTENT,
  SESSIONS_SIDEBAR_CONTENT,
  sessionMetaOf,
  settlePanel,
  swallowHiddenChrome,
  trackChrome,
} = await jiti.import("./embed/chrome.tsx");
const { LEARN_CHROME_KEYS, LEARN_PANEL_SCOPE } = await jiti.import("@gigachad/chat-ui/learn-protocol");
const sidebar = await jiti.import("./overrides/components/sidebar.tsx");
const topBar = await jiti.import("./overrides/components/top-bar.tsx");
const upstreamSidebar = await jiti.import("../components/sidebar.tsx");
const upstreamTopBar = await jiti.import("../components/top-bar.tsx");
const { toBlobs } = await jiti.import("@/components/activity-dock");
const { ServerEventsProvider } = await jiti.import("@/components/server-events");
const { ActivityProvider } = await jiti.import("@/components/activity-dock");
const { kbdEventForAction } = await jiti.import("@/src/core");

const UPDATED = "2026-10-29T10:00:00.000Z";
const chat = (id, fields = {}) => ({
  id,
  title: `Session ${id}`,
  kind: "teach",
  phase: "probe",
  busy: false,
  compacting: false,
  pendingQuestion: false,
  pendingProblem: false,
  updatedAt: UPDATED,
  ...fields,
});
const upstreamSource = (path) => readFileSync(new URL(`../${path}`, import.meta.url), "utf8");

/** Render *element* on *pathname* inside the app router contexts upstream's hooks need. */
function renderAt(pathname, element) {
  const router = { push() {}, replace() {}, refresh() {}, back() {}, forward() {}, prefetch() {} };
  return renderToStaticMarkup(h(AppRouterContext.Provider, { value: router }, h(PathnameContext.Provider, { value: pathname }, element)));
}

test("the overrides keep every upstream export", () => {
  for (const [override, upstream] of [
    [sidebar, upstreamSidebar],
    [topBar, upstreamTopBar],
  ]) {
    for (const name of Object.keys(upstream)) assert.ok(name in override, `override lacks upstream export ${name}`);
  }
});

test("the sessions sidebar is the app shell marker's <aside> child with its new-session entry, the page stays inside", () => {
  const { AppShell, SidebarProvider } = sidebar;
  const html = renderAt(
    "/",
    h(ServerEventsProvider, null, h(SidebarProvider, null, h(ActivityProvider, null, h(AppShell, null, h("p", { id: "page" }, "page"))))),
  );
  assert.match(html, /<div data-cf-app-shell=""><aside[ >]/, "markSessionsSidebar looks at the marker's <aside> children");
  assert.ok(html.includes('data-testid="new-session"'), `the sidebar's content marker ${SESSIONS_SIDEBAR_CONTENT} is gone`);
  assert.equal((html.match(/data-cf-app-shell/g) ?? []).length, 1);
  assert.equal((html.match(/<aside[ >]/g) ?? []).length, 1);
  assert.ok(html.includes('<p id="page">page</p>'));
});

test("the sidebar override stays a shared module that wraps nothing itself", () => {
  const source = readFileSync(new URL("./overrides/components/sidebar.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(source, /^\s*["']use client["']/m, "the root layout (a server component) imports its star exports");
  assert.doesNotMatch(source, /<\w/, "client code belongs in embed/chrome.tsx, a wrapper here renders twice");
});

test("the sidebar toggle renders inside its marker", () => {
  assert.match(renderAt("/", h(sidebar.SidebarToggle)), /^<span data-cf-sidebar-toggle=""><span /);
});

/** An element with *children*, each `{tagName, testids}`, recording the attributes set on them. */
function fakeShell(children) {
  return {
    children: children.map(({ tagName, testids }) => ({
      tagName,
      attributes: {},
      querySelector: (selector) => (testids.some((testid) => selector === `[data-testid="${testid}"]`) ? {} : null),
      setAttribute(name, value) {
        this.attributes[name] = value;
      },
    })),
  };
}

test("only the <aside> holding the new-session entry is marked as the sessions sidebar", () => {
  const shell = fakeShell([
    { tagName: "DIV", testids: ["new-session"] },
    { tagName: "ASIDE", testids: ["plan-panel"] },
    { tagName: "ASIDE", testids: ["new-session", "session-search"] },
  ]);
  assert.equal(markSessionsSidebar(shell), true);
  assert.deepEqual(shell.children.map((child) => child.attributes), [{}, {}, { "data-cf-sessions-sidebar": "" }]);
  assert.equal(markSessionsSidebar(fakeShell([{ tagName: "ASIDE", testids: [] }])), false, "a reshaped sidebar stays visible");
});

test("the top bar renders inside its marker, the session title bar is marked by its content", () => {
  const { TopBar } = topBar;
  for (const pathname of ["/chat/abc", "/", "/chat/new"]) {
    assert.match(renderAt(pathname, h(TopBar, null, "Title")), /^<div data-cf-top-bar=""><header /);
  }
  const bar = (content) => {
    const attributes = new Set();
    return {
      attributes,
      querySelector: (selector) => (selector === content ? {} : null),
      toggleAttribute: (name, force) => (force ? attributes.add(name) : attributes.delete(name)),
    };
  };
  const titleBar = bar(SESSION_TITLE_CONTENT);
  assert.equal(markSessionTitleBar(titleBar, "abc"), true);
  assert.deepEqual([...titleBar.attributes], ["data-cf-session-title"]);
  assert.equal(markSessionTitleBar(titleBar, null), false, "no session page, no title bar");
  assert.deepEqual([...titleBar.attributes], []);
  const chains = bar('[data-testid="chain-search"]');
  assert.equal(markSessionTitleBar(chains, "abc"), false, "another top bar on a session page stays visible");
});

test("upstream still renders the content the overlay keys its markers and panels to", () => {
  const chatSource = upstreamSource("components/chat.tsx");
  for (const testid of ["phase-chip", "plan-chip"]) assert.ok(chatSource.includes(`data-testid="${testid}"`), testid);
  assert.equal(SESSION_TITLE_CONTENT, '[data-testid="phase-chip"]');
  assert.ok(upstreamSource("components/sidebar.tsx").includes('data-testid="new-session"'));
});

test("chrome.css hides only in drive mode and only through the overrides' markers", () => {
  const css = readFileSync(new URL("./theme/chrome.css", import.meta.url), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
  const rules = [...css.matchAll(/([^{}]+)\{([^}]*)\}/g)].map(([, selectors, body]) => ({
    selectors: selectors.split(",").map((selector) => selector.trim()),
    body,
  }));
  assert.ok(rules.length > 0);
  for (const { selectors, body } of rules) {
    for (const selector of selectors) {
      assert.match(selector, /^(html\.cf-shell-drive )?\[data-cf-[a-z-]+\]/, selector);
      assert.doesNotMatch(selector, /\.(?!cf-shell-drive)[\w-]+/, `no upstream class names: ${selector}`);
      if (/display:\s*none/.test(body)) assert.match(selector, /^html\.cf-shell-drive /, selector);
    }
  }
});

test("session-meta carries the title bar's title, kind and phase, practice without a phase, and the row's state", () => {
  assert.deepEqual(sessionMetaOf(chat("a", { phase: "teach" })), {
    type: "session-meta",
    sessionId: "a",
    title: "Session a",
    kind: "teach",
    phase: "teach",
    status: null,
    updatedAt: UPDATED,
  });
  assert.deepEqual(sessionMetaOf(chat("b", { kind: "practice", phase: "probe" })), {
    type: "session-meta",
    sessionId: "b",
    title: "Session b",
    kind: "practice",
    phase: null,
    status: null,
    updatedAt: UPDATED,
  });
  assert.equal(sessionMetaOf(chat("c", { kind: undefined })).kind, "teach");
});

test("the row state follows upstream's order: compacting, waiting question, waiting problem (practice), working", () => {
  const status = (fields) => sessionMetaOf(chat("s", fields)).status;
  assert.equal(status({ compacting: true, pendingQuestion: true, busy: true }), "compacting");
  assert.equal(status({ pendingQuestion: true, busy: true }), "waiting");
  assert.equal(status({ kind: "practice", pendingProblem: true, busy: true }), "problem");
  assert.equal(status({ pendingProblem: true }), null, "a teaching session has no problem");
  assert.equal(status({ busy: true }), "working");
});

test("only changed session metas are posted again", () => {
  const posted = new Map();
  assert.deepEqual(changedMetas(posted, [chat("a"), chat("b")]).map((meta) => meta.sessionId), ["a", "b"]);
  assert.deepEqual(changedMetas(posted, [chat("a"), chat("b", { messageCount: 4 })]), []);
  assert.deepEqual(changedMetas(posted, [chat("a"), chat("b", { busy: true })]).map(({ sessionId, status }) => [sessionId, status]), [["b", "working"]]);
  assert.deepEqual(
    changedMetas(posted, [chat("a", { title: "Renamed" }), chat("b", { phase: "plan", busy: true })]).map(({ sessionId, title, phase }) => [sessionId, title, phase]),
    [
      ["a", "Renamed", "probe"],
      ["b", "Session b", "plan"],
    ],
  );
});

test("the meta poster reads the list once at a time, catches up after, and posts same-tab renames at once", async () => {
  const posts = [];
  let list = [chat("a")];
  let release;
  let reads = 0;
  const load = async () => {
    reads += 1;
    if (reads === 1) await new Promise((resolve) => (release = resolve));
    return list;
  };
  const poster = createSessionMetaPoster((meta) => posts.push(meta), load);

  const first = poster.refresh();
  list = [chat("a", { title: "Later" })];
  await poster.refresh();
  assert.equal(reads, 1, "a refresh during a read waits for it");
  release();
  await first;
  assert.equal(reads, 2);
  assert.deepEqual(posts.map((meta) => meta.title), ["Later"]);

  poster.renamed("a", "Typed");
  poster.renamed("a", "Typed");
  poster.renamed("unknown", "Ignored");
  assert.deepEqual(posts.map((meta) => meta.title), ["Later", "Typed"]);

  const failing = createSessionMetaPoster((meta) => posts.push(meta), async () => {
    throw new Error("offline");
  });
  await failing.refresh();
  assert.equal(posts.length, 2);
});

test("sessions a read no longer lists are reported once as removed", async () => {
  const posted = new Map();
  changedMetas(posted, [chat("a"), chat("b")]);
  assert.deepEqual(removedSessions(posted, [chat("b")]), ["a"]);
  assert.deepEqual(removedSessions(posted, [chat("b")]), []);

  const removed = [];
  let list = [chat("a"), chat("b")];
  const poster = createSessionMetaPoster(() => {}, async () => list, (ids) => removed.push(ids));
  await poster.refresh();
  assert.deepEqual(removed, [], "the first read removes nothing");
  list = [chat("b")];
  await poster.refresh();
  await poster.refresh();
  assert.deepEqual(removed, [["a"]]);
});

test("activity lists the background agents section's rows from upstream's feed", () => {
  const feed = {
    jobs: [
      { id: "j1", kind: "research", title: "Graphs", chatId: "c1", createdAt: "2026-10-29T10:00:00Z", activity: "Searching the web", contextLimit: null },
      { id: "j2", kind: "demo", title: "Graphs", chatId: "c1", createdAt: "2026-10-29T10:01:00Z", contextLimit: null },
    ],
    extractions: [{ id: "c2#1", chatId: "c2", title: "Sets", phase: "auditing", startedAt: Date.parse("2026-10-29T09:00:00Z") }],
  };
  assert.deepEqual(toBlobs(feed).map(activityItemOf), [
    { id: "c2#1", label: "Sets", status: "Extraction — auditing", sessionId: "c2" },
    { id: "j1", label: "Graphs", status: "Searching the web", sessionId: "c1" },
    { id: "j2", label: "Graphs", status: "Demo builder", sessionId: "c1" },
  ]);
});

/** A document with an optional settings dialog and modal, and MutationObservers the test triggers. */
function fakeChromeDocument() {
  const state = { nav: false, current: null, modal: false, clicked: [] };
  const landing = { click: () => state.clicked.push("agent") };
  const nav = { querySelector: (selector) => (selector === '[aria-current="true"]' && state.current ? {} : null) };
  const doc = {
    body: {},
    querySelector(selector) {
      if (selector === '[data-testid="settings-nav"]') return state.nav ? nav : null;
      if (selector === '[data-testid="settings-nav-agent"]') return state.nav ? landing : null;
      if (selector.includes('[aria-modal="true"]')) return state.modal ? {} : null;
      return null;
    },
  };
  return { doc, state };
}

test("a settings dialog showing no section moves to the landing section", () => {
  const { doc, state } = fakeChromeDocument();
  assert.equal(leaveUnknownSection(doc), false);
  state.nav = true;
  state.current = "graph";
  assert.equal(leaveUnknownSection(doc), false);
  state.current = null;
  assert.equal(leaveUnknownSection(doc), true);
  assert.deepEqual(state.clicked, ["agent"]);
});

test("modal is posted on mount and on every change, and the observer stops", () => {
  const observers = [];
  globalThis.MutationObserver = class {
    constructor(callback) {
      this.callback = callback;
      this.connected = false;
      observers.push(this);
    }
    observe() {
      this.connected = true;
    }
    disconnect() {
      this.connected = false;
    }
  };
  const { doc, state } = fakeChromeDocument();
  const posts = [];
  const stop = trackChrome(doc, (message) => posts.push(message));
  const [observer] = observers;
  state.modal = true;
  observer.callback();
  observer.callback();
  state.modal = false;
  observer.callback();
  assert.deepEqual(posts, [
    { type: "modal", open: false },
    { type: "modal", open: true },
    { type: "modal", open: false },
  ]);
  stop();
  assert.equal(observer.connected, false);
  delete globalThis.MutationObserver;
});

test("close-modal presses Escape on the open modal dialog, like Escape inside it, and does nothing without one", () => {
  const pressed = [];
  const dialog = { dispatchEvent: (event) => pressed.push({ type: event.type, key: event.key, bubbles: event.bubbles }) };
  let open = false;
  const doc = { querySelector: (selector) => (open && selector.includes('[role="dialog"][aria-modal="true"]') ? dialog : null) };
  const hadKeyboardEvent = "KeyboardEvent" in globalThis;
  if (!hadKeyboardEvent) {
    globalThis.KeyboardEvent = class extends Event {
      constructor(type, init) {
        super(type, init);
        this.key = init.key;
      }
    };
  }
  closeModal(doc);
  open = true;
  closeModal(doc);
  assert.deepEqual(pressed, [{ type: "keydown", key: "Escape", bubbles: true }]);
  if (!hadKeyboardEvent) delete globalThis.KeyboardEvent;
});

/** A window dispatching events at itself like a browser at the target: capture listeners first, then the others. */
function fakeWindow() {
  const listeners = [];
  return {
    listeners,
    addEventListener: (type, listener, options) => listeners.push({ type, listener, capture: Boolean(options?.capture) }),
    removeEventListener: (type, listener, options) => {
      const index = listeners.findIndex((entry) => entry.type === type && entry.listener === listener && entry.capture === Boolean(options?.capture));
      if (index >= 0) listeners.splice(index, 1);
    },
    dispatch(type) {
      let stopped = false;
      const event = { type, stopImmediatePropagation: () => (stopped = true) };
      for (const entry of [...listeners.filter((e) => e.capture), ...listeners.filter((e) => !e.capture)]) {
        if (stopped) break;
        if (entry.type === type) entry.listener(event);
      }
    },
  };
}

test("the hidden chrome's shortcuts go to the shell in drive mode only", () => {
  assert.deepEqual(HIDDEN_CHROME_ACTIONS, LEARN_CHROME_KEYS);
  const win = fakeWindow();
  const heard = [];
  const posts = [];
  const events = [...HIDDEN_CHROME_ACTIONS.map(kbdEventForAction), kbdEventForAction("open-settings")];
  for (const type of events) win.addEventListener(type, () => heard.push(type));
  let driven = true;
  const stop = swallowHiddenChrome(win, () => driven, (message) => posts.push(message));

  for (const type of events) win.dispatch(type);
  assert.deepEqual(heard, ["lh:kbd-open-settings"]);
  assert.deepEqual(posts, LEARN_CHROME_KEYS.map((action) => ({ type: "chrome-key", action })));

  driven = false;
  win.dispatch("lh:kbd-toggle-sidebar");
  assert.deepEqual(heard.at(-1), "lh:kbd-toggle-sidebar");
  assert.equal(posts.length, LEARN_CHROME_KEYS.length);

  stop();
  assert.equal(win.listeners.filter((entry) => entry.capture).length, 0);
});

test("every panel has a host, a request waits for its host and a newer one replaces it", () => {
  assert.deepEqual(Object.keys(PANEL_HOST).sort(), Object.keys(LEARN_PANEL_SCOPE).sort());
  requestPanel({ type: "open-panel", panel: "plan", sessionId: "s1" }, 1000);
  const plan = currentPanel("title-bar");
  assert.equal(plan?.sessionId, "s1");
  assert.equal(currentPanel("chrome-panel"), null);
  assert.equal(panelLive(plan, 1000 + PANEL_WAIT_MS), true);
  assert.equal(panelLive(plan, 1001 + PANEL_WAIT_MS), false, "a page that never came drops the request");

  requestPanel({ type: "open-panel", panel: "activity", sessionId: null });
  settlePanel(plan);
  assert.equal(currentPanel("title-bar"), null);
  const activity = currentPanel("chrome-panel");
  assert.equal(activity?.panel, "activity", "settling an older request keeps the newer one");
  settlePanel(activity);
  assert.equal(currentPanel("chrome-panel"), null);
});
